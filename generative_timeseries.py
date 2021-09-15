import os
import pickle
import functools
import tensorflow as tf
import tensorflow_probability as tfp

from toy_data import generate_2d_data
import surrogate_posteriors
from plot_utils import plot_heatmap_2d, plot_samples

import numpy as np
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root

num_iterations = int(1e3)

time_step_dim = 3
series_len = 30

@tfd.JointDistributionCoroutine
def lorenz_system():
  truth = []
  innovation_noise = .1
  step_size = 0.02
  loc = yield Root(tfd.Sample(tfd.Normal(0., 1., name='x_0'), sample_shape=3))
  for t in range(1, 30):
    x, y, z = tf.unstack(loc, axis=-1)
    truth.append(x)
    dx = 10 * (y - x)
    dy = x * (28 - z) - y
    dz = x * y - 8 / 3 * z
    delta = tf.stack([dx, dy, dz], axis=-1)
    loc = yield tfd.Independent(
      tfd.Normal(loc + step_size * delta,
                 tf.sqrt(step_size) * innovation_noise, name=f'x_{t}'),
      reinterpreted_batch_ndims=1)

def time_series_gen(batch_size):
  while True:
    yield lorenz_system.sample(batch_size)

def train(model, name, save_dir):

  def optimizer_step(net, inputs):
    with tf.GradientTape() as tape:
      loss = -net.log_prob(inputs)
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss

  def build_model(model_name):
    if model=='maf':
      scales = [tf.ones(time_step_dim) for _ in range(series_len)]
    else:
      scales = [tfp.util.TransformedVariable(tf.ones(time_step_dim), tfb.Softplus())
              for _ in range(series_len)]

    @tfd.JointDistributionCoroutine
    def prior_structure():
      new = yield Root(tfd.Independent(tfd.Normal(loc=0.,
                                  scale=scales[0]),1))

      for t in range(1, series_len):
        new = yield tfd.Independent(tfd.Normal(loc=new,
                               scale=scales[t]), 1)

    prior_matching_bijector = tfb.Chain(
      surrogate_posteriors._get_prior_matching_bijectors_and_event_dims(
        prior_structure)[-1])

    if model_name == 'maf':
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, 'maf')
    elif model_name == 'np_maf':
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure,
                                                         'normalizing_program',
                                                         'maf')
    maf.log_prob(prior_structure.sample(1))

    return maf, prior_matching_bijector

  maf, prior_matching_bijector = build_model(model)


  dataset = tf.data.Dataset.from_generator(functools.partial(time_series_gen, batch_size=int(100)),
                                             output_types=tf.float32).prefetch(tf.data.AUTOTUNE)

  lr = 1e-4
  lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=lr, decay_steps=num_iterations)
  optimizer = tf.optimizers.Adam(learning_rate=lr_decayed_fn)
  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                   weights=maf.trainable_variables)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, f'/tmp/{model}/tf_ckpts',
                                                  max_to_keep=20)
  train_loss_results = []

  for it in range(num_iterations):
    epoch_loss_avg = tf.keras.metrics.Mean()
    x = next(iter(dataset))

    # Optimize the model
    loss_value = optimizer_step(maf, x)
    # print(loss_value)
    epoch_loss_avg.update_state(loss_value)

    if it == 0:
      train_loss_results.append(epoch_loss_avg.result())
      best_loss = train_loss_results[-1]
    elif it % 10 == 0:
      train_loss_results.append(epoch_loss_avg.result())
      print(train_loss_results[-1])
      if tf.math.is_nan(train_loss_results[-1]):
        break
      if best_loss > train_loss_results[-1]:
        save_path = checkpoint_manager.save()
        best_loss = train_loss_results[-1]

  new_maf, _ = build_model(model)
  new_optimizer = tf.optimizers.Adam(learning_rate=lr)

  new_checkpoint = tf.train.Checkpoint(optimizer=new_optimizer,
                                       weights=new_maf.trainable_variables)
  new_checkpoint.restore(tf.train.latest_checkpoint(f'/tmp/{name}/tf_ckpts'))

  plt.plot(train_loss_results)
  plt.savefig(f'{save_dir}/loss_{name}.png',
              format="png")
  plt.close()

  if model == 'np_maf':
    for i in range(len(new_maf.distribution.bijector.bijectors)):
      if 'batch_normalization' in new_maf.distribution.bijector.bijectors[
        i].name:
        new_maf.distribution.bijector.bijectors[
          i].bijector.batchnorm.trainable = False

  results = new_maf.sample(100)
  with open(f'{save_dir}/samples/{name}.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


  print(f'{name} done!')
models = ['maf', 'np_maf']

main_dir = 'time_series_results'
if not os.path.isdir(main_dir):
  os.makedirs(main_dir)

for model in models:
  if model == 'maf':
    name = 'maf'
    train(model, name, save_dir=f'{main_dir}')
  elif model == 'rqs_maf':
    name = 'rqs_maf'
    for nbins in [8, 128]:
      train(model, name, save_dir=f'{main_dir}')
  else:
    for n_components in [100]:
      name = f'c{n_components}_{model}'
      train(model, name, save_dir=f'{main_dir}')