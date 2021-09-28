import os
import shutil
import pickle
import functools
import tensorflow as tf
import tensorflow_probability as tfp
import surrogate_posteriors
import timeseries_datasets
import process_stock

import numpy as np
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root

num_iterations = int(1e2)

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

def train(model, name, structure, dataset_name, save_dir):

  @tf.function
  def optimizer_step(net, inputs):
    with tf.GradientTape() as tape:
      loss = -net.log_prob(inputs)
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss

  @tf.function
  def eval(model, inputs):
    return -model.log_prob(inputs)

  if dataset_name == 'co2':
    time_step_dim = 1
    series_len = 24

  elif dataset_name == 'stock':
    series_len = 40
    time_step_dim = 1

  def build_model(model_name):
    if model=='maf':
      scales = tf.ones(time_step_dim)
    else:
      scales = tf.ones(time_step_dim)
      #scales = tfp.util.TransformedVariable(tf.ones(time_step_dim), tfb.Softplus())
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

    elif structure == 'stock':
      if model == 'maf':
        mul = 1.
        theta = 1.
        scale = 1.
      else:

        mul = tfp.util.TransformedVariable(1., tfb.Softplus())
        theta = tfp.util.TransformedVariable(1., tfb.Softplus())
        scale = tfp.util.TransformedVariable(1., tfb.Softplus())


      @tfd.JointDistributionCoroutine
      def prior_structure():
        x = yield Root(tfd.Normal(loc=0., scale=1., name='x_0'))
        v = yield Root(tfd.Normal(loc=0., scale=1., name='v_0'))
        for t in range(1, series_len):
          x = yield tfd.Normal(loc=x, scale=tf.math.exp(v), name=f'x_{t}')
          v = yield tfd.Normal(loc=mul*(v-theta), scale=scale, name=f'v_{t}')


    prior_matching_bijector = tfb.Chain(
      surrogate_posteriors._get_prior_matching_bijectors_and_event_dims(
        prior_structure)[-1])

    flow_params = {'num_hidden_units': 512}
    if model_name == 'maf':
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, 'maf', flow_params=flow_params)
    elif model_name == 'np_maf':
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure,
                                                         'gated_normalizing_program',
                                                         'maf',
                                                         flow_params=flow_params)
    elif model_name == 'bottom':
      maf = surrogate_posteriors.bottom_np_maf(prior_structure, flow_params)

    elif model_name == 'sandwich':
      maf = surrogate_posteriors._sandwich_maf_normalizing_program(
        prior_structure)

    maf.log_prob(prior_structure.sample(1))

    return maf, prior_matching_bijector

  maf, prior_matching_bijector = build_model(model)


  if dataset_name == 'co2':
    train, valid, test = timeseries_datasets.load_mauna_loa_atmospheric_co2()
    batch_size = 32
  elif dataset_name == 'stock':
    batch_size = 128
    train, valid, test = process_stock.get_stock_data()
    train = tf.reshape(train, [tf.shape(train)[0], -1])
    valid = tf.reshape(valid, [tf.shape(valid)[0], -1])
    test = tf.reshape(test, [tf.shape(test)[0], -1])

  train = tf.data.Dataset.from_tensor_slices(train).map(prior_matching_bijector).cache().shuffle(int(1e4)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
  valid = tf.data.Dataset.from_tensor_slices(valid).map(prior_matching_bijector).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
  test = tf.data.Dataset.from_tensor_slices(test).map(
    prior_matching_bijector).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
  lr = 1e-4
  lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=lr, decay_steps=1000)
  optimizer = tf.optimizers.Adam(learning_rate=lr_decayed_fn)
  checkpoint = tf.train.Checkpoint(weights=maf.trainable_variables)
  ckpt_dir = f'/tmp/{save_dir}/checkpoints/{name}'
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, ckpt_dir,
                                                  max_to_keep=20)
  train_loss_results = []
  valid_loss_results = []


  counter = 0
  for it in range(num_iterations):
    counter +=1
    train_loss_avg = tf.keras.metrics.Mean()
    for x in train:
      # Optimize the model
      loss_value = optimizer_step(maf, x)
      #print(loss_value)
      train_loss_avg.update_state(loss_value)

    train_loss_results.append(train_loss_avg.result())
    if tf.math.is_nan(train_loss_results[-1]):
      break

    valid_loss_avg = tf.keras.metrics.Mean()
    for x in valid:
      loss_value = eval(maf, x)
      valid_loss_avg.update_state(loss_value)

    valid_loss_results.append(valid_loss_avg.result())

    if it == 0:
      best_loss = valid_loss_avg.result()

    elif best_loss > valid_loss_avg.result():
      save_path = checkpoint_manager.save()
      best_loss = valid_loss_avg.result()

  new_maf, _ = build_model(model)

  new_checkpoint = tf.train.Checkpoint(weights=new_maf.trainable_variables)
  new_checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
  if os.path.isdir(f'{save_dir}/checkpoints/{name}'):
    clear_folder(f'{save_dir}/checkpoints/{name}')
  checkpoint_manager = tf.train.CheckpointManager(new_checkpoint,
                                                  f'{save_dir}/checkpoints/{name}',
                                                  max_to_keep=20)
  save_path = checkpoint_manager.save()

  plt.plot(train_loss_results)
  plt.plot(valid_loss_results)
  plt.savefig(f'{save_dir}/loss_{name}.png',
              format="png")
  plt.close()

  test_loss_avg = tf.keras.metrics.Mean()
  for x in test:
    loss_value = eval(maf, x)
    test_loss_avg.update_state(loss_value)

  results = {'samples' : tf.convert_to_tensor(new_maf.sample(1000)),
             'loss_eval': test_loss_avg.result(),
             'train_loss': train_loss_results,
             'valid_loss': valid_loss_results
             }
  with open(f'{save_dir}/{name}.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


  print(f'{name} done!')
models = ['maf', 'maf'] # 'sandwich']

main_dir = 'time_series_results'
if not os.path.isdir(main_dir):
  os.makedirs(main_dir)

datasets = ['stock']
n_runs = 5

for run in range(n_runs):

  for data in datasets:
    if not os.path.exists(f'{main_dir}/run_{run}/{data}'):
      os.makedirs(f'{main_dir}/run_{run}/{data}')
    for model in models:
      if model == 'maf':
        name = 'maf'
        train(model, name, structure='stock', dataset_name=data, save_dir=f'{main_dir}/run_{run}/{data}')
      elif model == 'bottom':
        name = 'bottom'
        train(model, name, structure='continuity', dataset_name=data,
              save_dir=f'{main_dir}/run_{run}/{data}')
      else:
        for structure in ['stock']: #, 'smoothness']:
          name = f'{model}_{structure}'
          train(model, name, structure, dataset_name=data, save_dir=f'{main_dir}/run_{run}/{data}')