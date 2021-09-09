import os
import pickle

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

num_epochs = 100
n = int(1e6)
n_dims = 2

def train(model, n_components, X, name, save_dir):

  def build_model(model_name):
    if model_name == 'maf':
      component_logits = tf.convert_to_tensor(
        [[1. / n_components for _ in range(n_components)] for _ in
         range(n_dims)])
      locs = tf.convert_to_tensor(
        [tf.linspace(-n_components / 2, n_components / 2, n_components) for _ in
         range(n_dims)])
      scales = tf.convert_to_tensor([[1. for _ in range(n_components)] for _ in
                                     range(n_dims)])
    else:
      component_logits = tf.Variable(
        [[1. / n_components for _ in range(n_components)] for _ in
         range(n_dims)])
      locs = tf.Variable(
        [tf.linspace(-4., 4., n_components) for _ in range(n_dims)])
      scales = tfp.util.TransformedVariable(
        [[1. for _ in range(n_components)] for _ in
         range(n_dims)], tfb.Softplus())

    @tfd.JointDistributionCoroutine
    def prior_structure():
      yield Root(tfd.Independent(tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=component_logits),
        components_distribution=tfd.Normal(loc=locs, scale=scales),
        name=f"prior"), 1))

    prior_matching_bijector = tfb.Chain(
      surrogate_posteriors._get_prior_matching_bijectors_and_event_dims(
        prior_structure)[-1])
    if model_name == 'maf':
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, 'maf')
    elif model_name == 'np_maf':
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, 'normalizing_program', 'maf')
    elif model_name == 'sandwich':
      maf = surrogate_posteriors._sandwich_maf_normalizing_program(prior_structure)

    maf.log_prob(prior_structure.sample(1))

    return maf, prior_matching_bijector

  @tf.function
  def optimizer_step(inputs):
    with tf.GradientTape() as tape:
      loss = -maf.log_prob(inputs)
    grads = tape.gradient(loss, maf.trainable_variables)
    optimizer.apply_gradients(zip(grads, maf.trainable_variables))
    return loss

  maf, prior_matching_bijector = build_model(model)

  X_train = prior_matching_bijector(X)
  dataset = tf.data.Dataset.from_tensor_slices(X_train)
  dataset = dataset.shuffle(2048, reshuffle_each_iteration=True).padded_batch(
    128)

  optimizer = tf.optimizers.Adam(learning_rate=1e-4)
  train_loss_results = []

  for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    for x in dataset:
      # Optimize the model
      loss_value = optimizer_step(x)
      epoch_loss_avg.update_state(loss_value)

    train_loss_results.append(epoch_loss_avg.result())

  plt.plot(train_loss_results)
  plt.savefig(f'{save_dir}/loss_{name}.png',
              format="png")
  plt.close()

  if model in ['np_maf', 'sandwich']:
    if model == 'np_maf':
      for i in range(len(maf.distribution.bijector.bijectors)):
        if 'batch_normalization' in maf.distribution.bijector.bijectors[i].name:
          maf.distribution.bijector.bijectors[i].bijector.batchnorm.trainable = False
    else:
      for i in range(len(maf.bijector.bijectors)):
        if 'batch_normalization' in maf.bijector.bijectors[i].name == 'batch_normalization':
          maf.bijector.bijectors[i].bijector.batchnorm.trainable = False

  plot_heatmap_2d(maf, matching_bijector=prior_matching_bijector,
                  mesh_count=500,
                  name=f'{save_dir}/density_{name}.png')
  plt.close()
  print(f'{name} done!')

datasets = ["8gaussians", "2spirals", 'checkerboard', "diamond"]
models = ['maf']

main_dir = '2d_toy_results'
if not os.path.isdir(main_dir):
  os.makedirs(main_dir)
for data in datasets:
  X, _ = generate_2d_data(data, batch_size=n)
  if not os.path.exists(f'{main_dir}/{data}'):
    os.makedirs(f'{main_dir}/{data}')
  plot_samples(X, name=f'{main_dir}/{data}/ground_truth.png')
  plt.close()
  for model in models:
    if model == 'maf':
      name = 'maf'
      train(model, 20, X, name, save_dir=f'{main_dir}/{data}')
    else:
      for n_components in [5, 100]:
        name = f'c{n_components}_{model}'
        train(model, n_components, X, name, save_dir=f'{main_dir}/{data}')