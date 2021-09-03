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
import surrogate_posteriors


tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tf.keras.layers
Root = tfd.JointDistributionCoroutine.Root

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
  low=-1,
  high=1
)


dist.network.load_weights(f'pcnn_weights/MNIST_{image_side_size}/')
seed = 15

def pixelcnn_as_jd(image_side_size=28,
                   num_observed_pixels=5, seed=None):

  @tf.function
  def forward_step(sampled_image, s, update_idxs):
    if tf.is_tensor(s):
      s = tf.clip_by_value(s, -1., 1.)
      if len(ps.shape(s)) <= 1:
        s = [s]
      sampled_image = tf.tensor_scatter_nd_update(sampled_image,
                                                  update_idxs,
                                                  tf.unstack(s))

    num_logistic_mix, locs, scales = dist.network(sampled_image)

    return sampled_image, num_logistic_mix, locs, scales


  def sample_channels(component_logits, locs, scales, row, col):
    if row == 0 and col == 0:
      return tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=component_logits[0, row, col]),
                                   components_distribution=tfd.Independent(tfd.Normal(loc=locs[0, row, col], scale=scales[0, row, col]), reinterpreted_batch_ndims=1),
                                   name=f"pixel_{row}_{col}")
    else:
      return tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=component_logits[:, row, col]),
                                   components_distribution=tfd.Independent(tfd.Normal(loc=locs[:, row, col], scale=scales[:, row, col]), reinterpreted_batch_ndims=1),
                                   name=f"pixel_{row}_{col}")

  @tfd.JointDistributionCoroutine
  def model() -> object:

    sampled_image = tf.zeros([1, image_side_size, image_side_size, 1])
    s = None
    batch_size = 0
    for i in range(image_side_size):
      for j in range(image_side_size):
        sampled_image, num_logistic_mix, locs, scales = forward_step(sampled_image, s, [[b, i, j] for b in
                                                   range(batch_size)])
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
        if ps.shape(s)[0] < batch_size:
          s = tf.repeat(s, batch_size, 0)
  '''@tfd.JointDistributionCoroutine
  def model() -> object:

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
                                                     range(batch_size)])'''

  ground_truth = model.sample(1, seed=seed)
  random.seed(seed)
  observations_idx = sorted(tf.math.top_k(tf.squeeze(ground_truth), k=num_observed_pixels, sorted=False).indices.numpy())  # assuming squared images
  observations = {f'pixel_{i//image_side_size}_{i%image_side_size}': ground_truth[i] for i in observations_idx}
  pixelcnn_prior = model.experimental_pin(**observations)
  observations = [tf.squeeze(o) for o in observations.values()]
  ground_truth_idx = [i for i in range(image_side_size ** 2) if
                      i not in observations_idx]

  def log_prob(samples):
    batch_size = ps.shape(samples[0])[0]
    s = tf.squeeze(tf.convert_to_tensor(samples), -1)
    o = tf.repeat(tf.expand_dims(tf.convert_to_tensor(observations), axis=1),
                  batch_size, 1)
    image_tensor = tf.reshape(tf.transpose(tf.clip_by_value(tf.dynamic_stitch([ground_truth_idx, observations_idx], [s,o]), -1,1), [1,0]), [-1, image_side_size, image_side_size])
    return dist.log_prob(tf.expand_dims(image_tensor, -1))

  return pixelcnn_prior, [ground_truth[i] for i in
                          ground_truth_idx], log_prob, observations, ground_truth_idx, observations_idx


prior, ground_truth, target_log_prob, observations,  ground_truth_idx, observations_idx = pixelcnn_as_jd(image_side_size=image_side_size, num_observed_pixels=5,
  seed=seed)

surrogate_posterior_name = 'normalizing_program'
backbone_posterior_name = 'iaf'
num_steps = 100
surrogate_posterior = get_surrogate_posterior(prior, surrogate_posterior_name,
                                              backbone_posterior_name)
surrogate_posterior.sample()
trainable_variables = list(surrogate_posterior.trainable_variables)
trainable_variables.extend(surrogate_posteriors.residual_fraction_vars.values())
print(surrogate_posteriors.residual_fraction_vars)
dist.network.trainable = False
start = time.time()
losses = tfp.vi.fit_surrogate_posterior(target_log_prob,
                                        surrogate_posterior,
                                        optimizer=tf.keras.optimizers.Adam(
                                        learning_rate=5e-5),
                                        # , gradient_transformers=[scale_grad_by_factor]),
                                        num_steps=num_steps,
                                        sample_size=10,
                                        trainable_variables=trainable_variables)
print(surrogate_posteriors.residual_fraction_vars)

print(f'Time taken: {time.time()-start}')
'''plt.plot(losses)
plt.show()'''
elbo = negative_elbo(target_log_prob, surrogate_posterior, num_samples=10,
                     seed=seed)
fkl = forward_kl(surrogate_posterior, ground_truth)
print(f'ELBO: {elbo}')
print(f'FORWARD_KL: {fkl}')


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
