import os
import shutil
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

num_iterations = int(5e5)
n_dims = 2

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

@tf.function
def sample(model, model_fixed, n_samples):
  x = model_fixed.distribution.sample(int(n_samples))
  results = {'initial_samples': x}

  for i in reversed(range(1, len(model_fixed.bijector.bijectors))):
    bij_name = model_fixed.bijector.bijectors[i].name
    if 'chain' in bij_name:
      x = model_fixed.bijector.bijectors[i].forward(x)
      results['inverse_mixture'] = x

    else:
      x = model.bijector.bijectors[i].forward(x)
      results[f'{bij_name}'] = x

  x = tf.convert_to_tensor(model.bijector.bijectors[0].forward(x))
  results['prior_matching'] = x

  return results

def train(model, n_components, name, save_dir):
  def build_model(model_name, trainable_mixture=True, component_logits=None,
                  locs=None, scales=None):
    if trainable_mixture:
      if model_name == 'maf' or model_name=='rqs_maf':
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
        if model_name == 'np_maf':
          loc_range = 4.
          scale = 1.
        else:
          loc_range = 10.
          scale = 3.
        component_logits = tf.Variable(
          [[1. / n_components for _ in range(n_components)] for _ in
           range(n_dims)], name='component_logits')
        locs = tf.Variable(
          [tf.linspace(-loc_range, loc_range, n_components) for _ in range(n_dims)],
          name='locs')
        scales = tfp.util.TransformedVariable(
          [[scale for _ in range(n_components)] for _ in
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
                                                         'gated_normalizing_program',
                                                         'maf')
    elif model_name == 'sandwich':
      maf = surrogate_posteriors._sandwich_maf_normalizing_program(
        prior_structure)

    elif model_name == 'rqs':
      flow_params = {
      'num_flow_layers': 2,
      'nbins': 128
    }
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, surrogate_posterior_name='rqs', flow_params=flow_params)
      maf.sample(1)
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

  dataset = tf.data.Dataset.from_generator(functools.partial(generate_2d_data, data=data, batch_size=int(100)),
                                           output_types=tf.float32)
  dataset = dataset.map(prior_matching_bijector).prefetch(tf.data.AUTOTUNE)
  lr = 1e-4
  lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=lr, decay_steps=num_iterations)
  optimizer = tf.optimizers.Adam(learning_rate=lr_decayed_fn)
  checkpoint = tf.train.Checkpoint(weights=maf.trainable_variables)
  ckpt_dir = f'/tmp/{save_dir}/checkpoints/{name}'
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, ckpt_dir,
                                                  max_to_keep=20)
  train_loss_results = []
  epoch_loss_avg = tf.keras.metrics.Mean()
  it = 0
  for x in dataset:

    # Optimize the model
    loss_value = optimizer_step(maf, x)
    # print(loss_value)
    epoch_loss_avg.update_state(loss_value)
    '''if tf.math.is_nan(epoch_loss_avg.result()):
      a = 0'''
    if it==0:
      best_loss = epoch_loss_avg.result()
      epoch_loss_avg = tf.keras.metrics.Mean()
    elif it % 100 == 0:
      train_loss_results.append(epoch_loss_avg.result())
      if tf.math.is_nan(train_loss_results[-1]):
        break
      save_path = checkpoint_manager.save()
      epoch_loss_avg = tf.keras.metrics.Mean()
    if it >= num_iterations:
      break
    it+=1

  new_maf, _ = build_model(model)

  new_checkpoint = tf.train.Checkpoint(weights=new_maf.trainable_variables)

  new_checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  x = tf.linspace(-4., 4., 500)
  y = tf.linspace(-4., 4., 500)
  X, Y = tf.meshgrid(x, y)

  concatenated_mesh_coordinates = tf.transpose(
    tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])]))

  plot_heatmap_2d(new_maf, matching_bijector=prior_matching_bijector,
                  mesh_count=500,
                  name=f'2d_toy_results/density_0.png')
  plt.close()
  # concatenated_mesh_coordinates = prior_matching_bijector(concatenated_mesh_coordinates)
  x = concatenated_mesh_coordinates
  for i in range(2, 5):
    x = new_maf.bijector.bijectors[i].inverse(x)
  prob = new_maf.distribution.prob(x)

  fig = plt.figure(frameon=False)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(tf.transpose(tf.reshape(prob, (500, 500))),
            aspect="equal")
  fig.savefig('2d_toy_results/density_1.png', format="png")
  plt.close()
  x = concatenated_mesh_coordinates
  for i in range(3, 5):
    x = new_maf.bijector.bijectors[i].inverse(x)
  prob = new_maf.distribution.prob(x)

  fig = plt.figure(frameon=False)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(tf.transpose(tf.reshape(prob, (500, 500))),
            aspect="equal")
  fig.savefig('2d_toy_results/density_2.png', format="png")
  plt.close()

  x = concatenated_mesh_coordinates
  prob = new_maf.distribution.prob(x)

  fig = plt.figure(frameon=False)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(tf.transpose(tf.reshape(prob, (500, 500))),
            aspect="equal")
  fig.savefig('2d_toy_results/density_3.png', format="png")
  plt.close()

  print(f'{name} done!')

datasets = ['checkerboard']
models = ['sandwich']#, 'np_maf', 'maf']

main_dir = '2d_toy_results'
if not os.path.isdir(main_dir):
  os.makedirs(main_dir)
n_runs = 1

for run in range(n_runs):
  for data in datasets:
    if not os.path.exists(f'{main_dir}/run_{run}/{data}'):
      os.makedirs(f'{main_dir}/run_{run}/{data}')
    for model in models:
      if model == 'maf':
        name = 'maf'
        train(model, 20, name, save_dir=f'{main_dir}/run_{run}/{data}')
      elif model == 'rqs':
        name = 'rqs'
        train(model, 20, name, save_dir=f'{main_dir}/run_{run}/{data}')
      else:
        for n_components in [100]:
          name = f'c{n_components}_{model}'
          train(model, n_components, name, save_dir=f'{main_dir}/run_{run}/{data}')