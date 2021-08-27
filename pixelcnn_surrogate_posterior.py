import random
import os
import pickle
import time
import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static as ps

import pixelcnn_original
from metrics import negative_elbo, forward_kl
from surrogate_posteriors import get_surrogate_posterior


tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tf.keras.layers
Root = tfd.JointDistributionCoroutine.Root


def pixelcnn_as_jd(network, num_logistic_mix=5, image_side_size=28,
                   num_observed_pixels=5, dtype=tf.float32, seed=None):
  def sample_channels(component_logits, locs, scales, row, col):
    num_channels = 1  # so far working with 1 channel images
    component_dist = tfd.Categorical(logits=component_logits)
    mask = tf.one_hot(indices=component_dist.sample(seed=seed),
                      depth=num_logistic_mix)
    mask = tf.cast(mask[..., tf.newaxis], dtype)

    # apply mixture component mask and separate out RGB parameters
    masked_locs = tf.reduce_sum(locs * mask, axis=-2)
    loc_tensors = tf.split(masked_locs, num_channels, axis=-1)
    masked_scales = tf.reduce_sum(scales * mask, axis=-2)
    scale_tensors = tf.split(masked_scales, num_channels, axis=-1)
    if row == 0 and col == 0:
      return tfd.Independent(tfd.Normal(loc=loc_tensors[0][0, row, col],
                                        scale=scale_tensors[0][0, row, col],
                                        name=f"pixel_{row}_{col}"),
                             reinterpreted_batch_ndims=1)
    else:
      return tfd.Independent(tfd.Normal(loc=loc_tensors[0][:, row, col],
                                        scale=scale_tensors[0][:, row, col],
                                        name=f"pixel_{row}_{col}"),
                             reinterpreted_batch_ndims=1)

  @tfd.JointDistributionCoroutine
  def model():

    sampled_image = tf.zeros([1, image_side_size, image_side_size, 1])
    for i in range(image_side_size):
      for j in range(image_side_size):
        num_logistic_mix, locs, scales = network(sampled_image)
        next_pixel = sample_channels(num_logistic_mix, locs, scales, row=i,
                                     col=j)
        if i == 0 and j == 0:
          s = yield Root(next_pixel)
          if len(ps.shape(s)) > 1:
            batch_size = ps.shape(s)[0]
            sampled_image = tf.repeat(sampled_image, batch_size, 0)
          else:
            batch_size = 1
        else:
          s = yield next_pixel
        s = tf.clip_by_value(s, -1., 1.)
        if len(ps.shape(s)) <= 1:
          s = [s]
        if ps.shape(s)[0] < batch_size:
          s = tf.repeat(s, batch_size, 0)
        sampled_image = tf.tensor_scatter_nd_update(sampled_image,
                                                    [[b, i, j] for b in
                                                     range(batch_size)],
                                                    [s[c] for c in
                                                     range(batch_size)])

  ground_truth = model.sample(1, seed=seed)
  random.seed(seed)
  observations_idx = sorted(random.sample(range(image_side_size ** 2),
                                          num_observed_pixels))  # assuming squared images
  observations = {f'var{i}': ground_truth[i] for i in observations_idx}
  pixelcnn_prior = model.experimental_pin(**observations)
  ground_truth_idx = [i for i in range(image_side_size ** 2) if
                      i not in observations_idx]

  return pixelcnn_prior, [ground_truth[i] for i in
                          ground_truth_idx], pixelcnn_prior.unnormalized_log_prob, observations, ground_truth_idx, observations_idx


image_side_size = 14
image_shape = (image_side_size, image_side_size, 1)

dist = pixelcnn_original.PixelCNN(
  image_shape=image_shape,
  num_resnet=1,
  num_hierarchies=2,
  num_filters=32,
  num_logistic_mix=5,
  dropout_p=.3,
  use_weight_norm=False,
)

dist.network.load_weights(f'pcnn_weights/MNIST_{image_side_size}/')
dist.network.trainable = False
samples = dist.sample(5)
seed = 20
prior, ground_truth, target_log_prob, observations,  ground_truth_idx, observations_idx = pixelcnn_as_jd(
  dist.network, image_side_size=image_side_size, num_observed_pixels=10,
  seed=seed)

surrogate_posterior_name = 'normalizing_program'
backbone_posterior_name = 'iaf'
num_steps = 1000
surrogate_posterior = get_surrogate_posterior(prior, surrogate_posterior_name,
                                              backbone_posterior_name)
start = time.time()
@tf.function(jit_compile=True) # this is useful
def fit_vi():
  return tfp.vi.fit_surrogate_posterior(target_log_prob,
                                        surrogate_posterior,
                                        optimizer=tf.keras.optimizers.Adam(
                                        learning_rate=5e-5),
                                        # , gradient_transformers=[scale_grad_by_factor]),
                                        num_steps=num_steps,
                                        sample_size=5)

losses = fit_vi()
print(f'Time taken: {time.time()-start}')
'''plt.plot(losses)
plt.show()'''
elbo = negative_elbo(target_log_prob, surrogate_posterior, num_samples=10,
                     seed=seed)
fkl = forward_kl(surrogate_posterior, ground_truth)
print(f'ELBO: {elbo}')
print(f'FORWARD_KL: {fkl}')


observations = [tf.squeeze(o) for o in observations.values()]
ground_truth = [tf.squeeze(g) for g in ground_truth]
samples = tf.convert_to_tensor(surrogate_posterior.sample(10))
results = {'loss': losses,
           'elbo': elbo,
           'fkl': fkl,
           'ground_truth': ground_truth,
           'observations': observations,
           'ground_truth_idx': ground_truth_idx,
           'observations_idx': observations_idx,
           'samples': samples}

if backbone_posterior_name:
  surrogate_posterior_name = f'{surrogate_posterior_name}_{backbone_posterior_name}'

repo_name = f'pixelcnn/{surrogate_posterior_name}'
if not os.path.exists(repo_name):
  os.makedirs(repo_name)

with open(f'{repo_name}/res.pickle', 'wb') as handle:
  pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
