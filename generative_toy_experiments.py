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

num_epochs = 50
n = int(1e5)
n_dims = 2

def train(model, n_components, X, name, save_dir):
  def build_model(model_name, trainable_mixture=True, component_logits=None,
                  locs=None, scales=None):
    if trainable_mixture:
      if model_name == 'maf':
        component_logits = tf.convert_to_tensor(
          [[1. / n_components for _ in range(n_components)] for _ in
           range(n_dims)])
        locs = tf.convert_to_tensor(
          [tf.linspace(-n_components / 2, n_components / 2, n_components) for _
           in
           range(n_dims)])
        scales = tf.convert_to_tensor(
          [[1. for _ in range(n_components)] for _ in
           range(n_dims)])
      else:
        component_logits = tf.Variable(
          [[1. / n_components for _ in range(n_components)] for _ in
           range(n_dims)], name='component_logits')
        locs = tf.Variable(
          [tf.linspace(-4., 4., n_components) for _ in range(n_dims)],
          name='locs')
        scales = tfp.util.TransformedVariable(
          [[1. for _ in range(n_components)] for _ in
           range(n_dims)], tfb.Softplus(), name='scales')

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
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure,
                                                         'normalizing_program',
                                                         'maf')
    elif model_name == 'sandwich':
      maf = surrogate_posteriors._sandwich_maf_normalizing_program(
        prior_structure)

    maf.log_prob(prior_structure.sample(1))

    return maf, prior_matching_bijector

  @tf.function
  def optimizer_step(net, inputs):
    with tf.GradientTape() as tape:
      loss = -net.log_prob(inputs)
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
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
      loss_value = optimizer_step(maf, x)
      # print(loss_value)
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

  if model == 'sandwich':
    for v in maf.trainable_variables:
      if 'locs' in v.name:
        locs = tf.convert_to_tensor(v)
      elif 'scales' in v.name:
        scales = tf.convert_to_tensor(v)
      elif 'component_logits' in v.name:
        component_logits = tf.convert_to_tensor(v)

    fixed_maf, _ = build_model('sandwich', trainable_mixture=False,
                               component_logits=component_logits, locs=locs,
                               scales=scales)

    if not os.path.exists(f'{save_dir}/bijector_steps'):
      os.makedirs(f'{save_dir}/bijector_steps')
    x = fixed_maf.distribution.sample(int(1e4))
    plot_samples(x, name=f'{save_dir}/bijector_steps/initial_samples.png')
    plt.close()
    for i in reversed(range(1,len(fixed_maf.bijector.bijectors))):
      bij_name = fixed_maf.bijector.bijectors[i].name
      if 'chain' in bij_name:
        x = fixed_maf.bijector.bijectors[i].forward(x)
        plot_samples(x, npts=250, name=f'{save_dir}/bijector_steps/inverse_mixture.png')
      else:
        x = maf.bijector.bijectors[i].forward(x)
        plot_samples(x, npts=250, name=f'{save_dir}/bijector_steps/{bij_name}_{i}.png')
      plt.close()
  print(f'{name} done!')

datasets = ["8gaussians", "2spirals", 'checkerboard', "diamond"]
models = ['sandwich']

main_dir = '2d_toy_results'
if not os.path.isdir(main_dir):
  os.makedirs(main_dir)
for data in datasets:
  X, _ = generate_2d_data(data, batch_size=n)
  if not os.path.exists(f'{main_dir}/{data}'):
    os.makedirs(f'{main_dir}/{data}')
  '''plot_samples(X, npts=500, name=f'{main_dir}/{data}/ground_truth.png')
  plt.close()'''
  for model in models:
    if model == 'maf':
      name = 'maf'
      train(model, 20, X, name, save_dir=f'{main_dir}/{data}')
    else:
      for n_components in [100]:
        name = f'c{n_components}_{model}'
        train(model, n_components, X, name, save_dir=f'{main_dir}/{data}')