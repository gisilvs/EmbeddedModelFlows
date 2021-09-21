import os
import pickle
import functools
import tensorflow as tf
import tensorflow_probability as tfp
import surrogate_posteriors
import timeseries_datasets

import numpy as np
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root

num_iterations = int(5e4)

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
    series_len = 12

  def build_model(model_name):
    if model=='maf':
      scales = tf.ones(time_step_dim)
    else:
      scales = tfp.util.TransformedVariable(tf.ones(time_step_dim), tfb.Softplus())
    initial_mean = tf.zeros(time_step_dim)

    if structure == 'continuity':
      @tfd.JointDistributionCoroutine
      def prior_structure():
        new = yield Root(tfd.Independent(tfd.Normal(loc=initial_mean,
                                    scale=tf.ones_like(initial_mean)),1))

        for t in range(1, series_len):
          new = yield tfd.Independent(tfd.Normal(loc=new,
                                 scale=scales), 1)

    elif structure == 'smoothness':
      @tfd.JointDistributionCoroutine
      def prior_structure():
        previous = yield Root(tfd.Independent(tfd.Normal(loc=initial_mean,
                                                    scale=tf.ones_like(initial_mean)), 1))
        current = yield Root(tfd.Independent(tfd.Normal(loc=initial_mean,
                                                    scale=tf.ones_like(initial_mean)), 1))
        for t in range(2, series_len):
          new = yield tfd.Independent(tfd.Normal(loc=2 * current - previous,
                                                 scale=scales), 1)
          previous = current
          current = new

    prior_matching_bijector = tfb.Chain(
      surrogate_posteriors._get_prior_matching_bijectors_and_event_dims(
        prior_structure)[-1])

    if model_name == 'maf':
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, 'maf')
    elif model_name == 'np_maf':
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure,
                                                         'normalizing_program',
                                                         'maf')
    elif model_name == 'sandwich':
      maf = surrogate_posteriors._sandwich_maf_normalizing_program(
        prior_structure)

    maf.log_prob(prior_structure.sample(1))

    return maf, prior_matching_bijector

  maf, prior_matching_bijector = build_model(model)


  if dataset_name == 'co2':
    train, valid, test = timeseries_datasets.load_mauna_loa_atmospheric_co2()
    batch_size = 32
  train = tf.data.Dataset.from_tensor_slices(train).map(prior_matching_bijector).batch(batch_size).prefetch(tf.data.AUTOTUNE).shuffle(int(10e3))
  valid = tf.data.Dataset.from_tensor_slices(valid).map(prior_matching_bijector).batch(batch_size).prefetch(tf.data.AUTOTUNE).shuffle(int(10e3))
  test = tf.data.Dataset.from_tensor_slices(test).map(
    prior_matching_bijector).batch(batch_size).prefetch(tf.data.AUTOTUNE).shuffle(
    int(10e3))
  lr = 1e-4
  '''lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=lr, decay_steps=num_iterations)'''
  optimizer = tf.optimizers.Adam(learning_rate=lr)
  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                   weights=maf.trainable_variables)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, f'/tmp/{model}/tf_ckpts',
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

    valid_loss_avg = tf.keras.metrics.Mean()
    for x in valid:
      loss_value = eval(maf, x)
      valid_loss_avg.update_state(loss_value)

    valid_loss_results.append(valid_loss_avg.result())
    print(valid_loss_results[-1])
    print(counter)

    if it == 0:
      best_loss = valid_loss_avg.result()
      counter = 0

    elif best_loss > valid_loss_avg.result():
      save_path = checkpoint_manager.save()
      best_loss = valid_loss_avg.result()
      counter = 0

    elif counter >=30:
      break


  new_maf, _ = build_model(model)
  new_optimizer = tf.optimizers.Adam(learning_rate=lr)

  new_checkpoint = tf.train.Checkpoint(optimizer=new_optimizer,
                                       weights=new_maf.trainable_variables)
  new_checkpoint.restore(tf.train.latest_checkpoint(f'/tmp/{name}/tf_ckpts'))

  checkpoint_manager = tf.train.CheckpointManager(new_checkpoint,
                                                  f'{save_dir}/checkpoints/{name}',
                                                  max_to_keep=20)
  save_path = checkpoint_manager.save()

  plt.plot(train_loss_results)
  plt.savefig(f'{save_dir}/loss_{name}.png',
              format="png")
  plt.close()

  if model == 'np_maf':
    for i in range(len(new_maf.distribution.bijector.bijectors)):
      if 'batch_normalization' in new_maf.distribution.bijector.bijectors[
        i].name:
        new_maf.distribution.bijector.bijectors[
          i].batchnorm.trainable = False
  else:
    for i in range(len(new_maf.bijector.bijectors)):
      if 'batch_normalization' in new_maf.bijector.bijectors[
        i].name == 'batch_normalization':
        new_maf.bijector.bijectors[i].batchnorm.trainable = False

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
models = ['np_maf', 'maf'] # 'sandwich']

main_dir = 'time_series_results'
if not os.path.isdir(main_dir):
  os.makedirs(main_dir)

datasets = ['co2']
n_runs = 5

for run in range(n_runs):

  for data in datasets:
    if not os.path.exists(f'{main_dir}/run_{run}/{data}'):
      os.makedirs(f'{main_dir}/run_{run}/{data}')
    for model in models:
      if model == 'maf':
        name = 'maf'
        train(model, name, structure='continuity', dataset_name=data, save_dir=f'{main_dir}/run_{run}/{data}')
      elif model == 'rqs_maf':
        name = 'rqs_maf'
        for nbins in [8, 128]:
          train(model, name, save_dir=f'{main_dir}/run_{run}/{data}')
      else:
        for structure in ['continuity', 'smoothness']: #, 'smoothness']:
          name = f'{model}_{structure}'
          train(model, model, structure, dataset_name=data, save_dir=f'{main_dir}/run_{run}/{data}')