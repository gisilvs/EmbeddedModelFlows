import os
import shutil
import time
import pickle
import functools
import tensorflow as tf
import tensorflow_probability as tfp
import surrogate_posteriors

from tensorflow_probability.python.internal import prefer_static as ps
from toy_data import generate_2d_data
import surrogate_posteriors
from plot_utils import plot_heatmap_2d, plot_samples

import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root

num_iterations = int(4e5)


def clear_folder(folder):
  for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
      elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
    except Exception as e:
      print('Failed to delete %s. Reason: %s' % (file_path, e))


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


@tfd.JointDistributionCoroutine
def brownian_motion():
  new = yield Root(tfd.Normal(loc=0, scale=.1))

  for t in range(1, 30):
    new = yield tfd.Normal(loc=new, scale=.1)


@tfd.JointDistributionCoroutine
def ornstein_uhlenbeck():
  a = 0.8
  new = yield Root(tfd.Normal(loc=0, scale=5.))

  for t in range(1, 30):
    new = yield tfd.Normal(loc=a * new, scale=.5)


@tfd.JointDistributionCoroutine
def van_der_pol():
  mul = 4
  innovation_noise = .1
  mu = 1.
  step_size = 0.05
  loc = yield Root(tfd.Sample(tfd.Normal(0., 1., name='x_0'), sample_shape=2))
  for t in range(1, 30 * mul):
    x, y = tf.unstack(loc, axis=-1)
    dx = y
    dy = mu * (1 - x ** 2) * y - x
    delta = tf.stack([dx, dy], axis=-1)
    loc = yield tfd.Independent(
      tfd.Normal(loc + step_size * delta,
                 tf.sqrt(step_size) * innovation_noise, name=f'x_{t}'),
      reinterpreted_batch_ndims=1)


def time_series_gen(batch_size, dataset_name):
  if dataset_name == 'lorenz':
    while True:
      yield tf.reshape(
        tf.transpose(tf.convert_to_tensor(lorenz_system.sample(batch_size)),
                     [1, 0, 2]), [batch_size, -1])
  if dataset_name == 'lorenz_scaled':
    while True:
      samples = tf.convert_to_tensor(lorenz_system.sample(batch_size))
      std = tf.math.reduce_std(samples, axis=1)
      samples = samples / tf.expand_dims(std, 1)
      yield tf.reshape(tf.transpose(samples, [1, 0, 2]), [batch_size, -1])
  if dataset_name == 'van_der_pol':
    while True:
      yield tf.reshape(
        tf.transpose(tf.convert_to_tensor(van_der_pol.sample(batch_size)),
                     [1, 0, 2]), [batch_size, -1])
  elif dataset_name == 'brownian':
    while True:
      yield tf.math.exp(tf.reshape(
        tf.transpose(tf.convert_to_tensor(brownian_motion.sample(batch_size)),
                     [1, 0]), [batch_size, -1]))
  elif dataset_name == 'ornstein':
    while True:
      yield tf.reshape(tf.transpose(
        tf.convert_to_tensor(ornstein_uhlenbeck.sample(batch_size)), [1, 0]),
                       [batch_size, -1])


def train(model, name, structure, dataset_name, save_dir):
  @tf.function
  def optimizer_step(net, inputs):
    with tf.GradientTape() as tape:
      loss = -net.log_prob(inputs)
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss

  if dataset_name == 'lorenz' or dataset_name == 'lorenz_scaled':
    time_step_dim = 3
    series_len = 30

  elif dataset_name == 'brownian' or dataset_name == 'ornstein':
    time_step_dim = 1
    series_len = 30

  elif dataset_name == 'van_der_pol':
    time_step_dim = 2
    series_len = 120

  def build_model(model_name):
    if model == 'maf' or model == 'maf3' or model == 'maf_swap' or model == \
        'splines':
      scales = tf.ones(time_step_dim)
    else:
      scales = tfp.util.TransformedVariable(tf.ones(time_step_dim),
                                            tfb.Softplus())
    initial_mean = tf.zeros(time_step_dim)

    if structure == 'continuity':
      @tfd.JointDistributionCoroutine
      def prior_structure():
        new = yield Root(tfd.Independent(tfd.Normal(loc=initial_mean,
                                                    scale=tf.ones_like(
                                                      initial_mean),
                                                    name='prior0'), 1))

        for t in range(1, series_len):
          new = yield tfd.Independent(tfd.Normal(loc=new,
                                                 scale=scales,
                                                 name=f'prior{t}'), 1)

    elif structure == 'smoothness':
      @tfd.JointDistributionCoroutine
      def prior_structure():
        previous = yield Root(tfd.Independent(tfd.Normal(loc=initial_mean,
                                                         scale=tf.ones_like(
                                                           initial_mean),
                                                         name='prior0'), 1))
        current = yield Root(tfd.Independent(tfd.Normal(loc=initial_mean,
                                                        scale=tf.ones_like(
                                                          initial_mean),
                                                        name='prior1'), 1))
        for t in range(2, series_len):
          new = yield tfd.Independent(tfd.Normal(loc=2 * current - previous,
                                                 scale=scales,
                                                 name=f'prior{t}'), 1)
          previous = current
          current = new

    prior_matching_bijector = tfb.Chain(
      surrogate_posteriors._get_prior_matching_bijectors_and_event_dims(
        prior_structure)[-1])

    if model_name == 'maf':
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, 'maf')
    elif model_name == 'maf_swap':
      flow_params = {'swap': False}
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, 'maf',
                                                         flow_params=flow_params)
    elif model_name == 'maf3':
      flow_params = {'num_flow_layers': 3}
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, 'maf',
                                                         flow_params=flow_params)
    elif model_name == 'maf3_swap':
      flow_params = {'num_flow_layers': 3}
      flow_params['swap'] = False
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, 'maf',
                                                         flow_params=flow_params)
    elif model_name == 'np_maf':
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure,
                                                         'gated_normalizing_program',
                                                         'maf')
    elif model_name == 'splines':
      flow_params = {
        'layers': 6,
        'number_of_bins': 32,
        'input_dim': series_len * time_step_dim,
        'nn_layers': [32, 32],
        'b_interval': 10
      }
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure,
                                                         surrogate_posterior_name='splines',
                                                         flow_params=flow_params)
      maf.sample(2)
    elif model_name == 'np_splines':
      flow_params = {
        'layers': 6,
        'number_of_bins': 32,
        'input_dim': series_len * time_step_dim,
        'nn_layers': [32, 32],
        'b_interval': 10
      }
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure,
                                                         surrogate_posterior_name='gated_normalizing_program',
                                                         backnone_name='splines',
                                                         flow_params=flow_params)
      maf.sample(2)
    elif model_name == 'bottom':
      maf = surrogate_posteriors.bottom_np_maf(prior_structure)
    elif model_name == 'sandwich':
      maf = surrogate_posteriors._sandwich_maf_normalizing_program(
        prior_structure)

    maf.log_prob(prior_structure.sample(2))

    return maf, prior_matching_bijector

  maf, prior_matching_bijector = build_model(model)
  if 'splines' == model and dataset_name == 'lorenz':
    scale_bijector = tfb.Scale(tf.convert_to_tensor([7.5674453 for _ in range(
      30)] + [8.48064 for _ in range(
      30)] + [15.134891 for _ in range(
      30)]))
    maf = tfd.TransformedDistribution(
      distribution=maf,
      bijector=tfb.Chain([prior_matching_bijector, scale_bijector,
                          tfb.Invert(prior_matching_bijector)])
    )

  lr = 1e-4
  lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=lr, decay_steps=5e5)
  optimizer = tf.optimizers.Adam(learning_rate=lr_decayed_fn)
  checkpoint = tf.train.Checkpoint(weights=maf.trainable_variables)
  ckpt_dir = f'/tmp/{save_dir}/checkpoints/{name}'
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, ckpt_dir,
                                                  max_to_keep=20)
  train_loss_results = []

  epoch_loss_avg = tf.keras.metrics.Mean()
  it = 0
  dataset = tf.data.Dataset.from_generator(
    functools.partial(time_series_gen, batch_size=int(1e5),
                      dataset_name=dataset_name),
    output_types=tf.float32)# \
    #.map(prior_matching_bijector, num_parallel_calls=tf.data.AUTOTUNE) \
    #.prefetch(tf.data.AUTOTUNE)

  is_break = False
  #start = time.time()
  while it < num_iterations:
    if is_break:
      break
    train_data = next(iter(dataset))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)\
      .map(prior_matching_bijector, num_parallel_calls=tf.data.AUTOTUNE) \
      .batch(100).prefetch(tf.data.AUTOTUNE)
    for x in train_dataset:

      # Optimize the model
      loss_value = optimizer_step(maf, x)
      epoch_loss_avg.update_state(loss_value)

      if it == 0:
        best_loss = epoch_loss_avg.result()
        epoch_loss_avg = tf.keras.metrics.Mean()
        save_path = checkpoint_manager.save()
      elif it % 100 == 0:
        train_loss_results.append(epoch_loss_avg.result())
        epoch_loss_avg = tf.keras.metrics.Mean()
        if tf.math.is_nan(train_loss_results[-1]):
          is_break = True
          break
        else:
          save_path = checkpoint_manager.save()

      if it % 10000 == 0 and it > 0:
        #print(time.time()-start)
        #start = time.time()
        print(train_loss_results[-1])
        print(it)
      if it >= num_iterations:
        is_break = True
        break
      it += 1

  new_maf, _ = build_model(model)
  if 'splines' == model and dataset_name == 'lorenz':
    scale_bijector = tfb.Scale(tf.convert_to_tensor([7.5674453 for _ in range(
      30)] + [8.48064 for _ in range(
      30)] + [15.134891 for _ in range(
      30)]))
    new_maf = tfd.TransformedDistribution(
      distribution=new_maf,
      bijector=tfb.Chain([prior_matching_bijector, scale_bijector,
                          tfb.Invert(prior_matching_bijector)])
    )

  new_checkpoint = tf.train.Checkpoint(weights=new_maf.trainable_variables)
  new_checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
  if os.path.isdir(f'{save_dir}/checkpoints/{name}'):
    clear_folder(f'{save_dir}/checkpoints/{name}')
  checkpoint_manager = tf.train.CheckpointManager(new_checkpoint,
                                                  f'{save_dir}/checkpoints/{name}',
                                                  max_to_keep=20)
  save_path = checkpoint_manager.save()

  plt.plot(train_loss_results)
  plt.savefig(f'{save_dir}/loss_{name}.png',
              format="png")
  plt.close()

  eval_dataset = tf.data.Dataset.from_generator(functools.partial(
    time_series_gen, batch_size=int(1e4), dataset_name=dataset_name),
    output_types=tf.float32).map(prior_matching_bijector)

  eval_log_prob = -tf.reduce_mean(new_maf.log_prob(next(iter(eval_dataset))))

  results = {'samples': tf.convert_to_tensor(new_maf.sample(1000)),
             'loss_eval': eval_log_prob,
             'loss': train_loss_results
             }
  with open(f'{save_dir}/{name}.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

  print(f'{name} done!')


# maf_swap means that no swap is done
models = [
  #'np_maf',
  'np_splines',
  #'maf',
  #'maf3',
  #'splines',
  #'bottom'
]

main_dir = 'time_series_results'
if not os.path.isdir(main_dir):
  os.makedirs(main_dir)

datasets = ['lorenz']
n_runs = [2]

for run in n_runs:

  for data in datasets:
    if not os.path.exists(f'{main_dir}/run_{run}/{data}'):
      os.makedirs(f'{main_dir}/run_{run}/{data}')
    for model in models:
      if model == 'maf' or model == 'maf3' or model == 'maf_swap' or model == \
          'bottom' or model == 'maf3_swap' or model == 'splines':
        name = model
        train(model, name, structure='continuity', dataset_name=data,
              save_dir=f'{main_dir}/run_{run}/{data}')
      else:
        for structure in ['continuity']:
          name = f'{model}_{structure}'
          train(model, name, structure, dataset_name=data,
                save_dir=f'{main_dir}/run_{run}/{data}')