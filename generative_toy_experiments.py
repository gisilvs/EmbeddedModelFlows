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

num_iterations = int(100)
n_dims = 2

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
        else:
          loc_range = 4.
        component_logits = tf.Variable(
          [[1. / n_components for _ in range(n_components)] for _ in
           range(n_dims)], name='component_logits')
        locs = tf.Variable(
          [tf.linspace(-loc_range, loc_range, n_components) for _ in range(n_dims)],
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

    elif model_name == 'rqs_maf':
      flow_params = {
      'num_flow_layers': 2,
      'nbins': nbins
    }
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, surrogate_posterior_name='rqs_maf', flow_params=flow_params)
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
  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                   weights=maf.trainable_variables)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, f'/tmp/{name}/tf_ckpts',
                                                  max_to_keep=20)
  train_loss_results = []
  epoch_loss_avg = tf.keras.metrics.Mean()
  for it in range(num_iterations):

    x = next(iter(dataset))
    # Optimize the model
    loss_value = optimizer_step(maf, x)
    if np.array(tf.math.is_nan(loss_value)).any():
      a=0
    # print(loss_value)
    save_path = checkpoint_manager.save()
    epoch_loss_avg.update_state(loss_value)
    if it==0:
      best_loss = epoch_loss_avg.result()
      epoch_loss_avg = tf.keras.metrics.Mean()
    elif it % 100 == 0:
      train_loss_results.append(epoch_loss_avg.result())
      if tf.math.is_nan(train_loss_results[-1]):
        break
      if best_loss > train_loss_results[-1]:
        save_path = checkpoint_manager.save()
        best_loss = train_loss_results[-1]
      epoch_loss_avg = tf.keras.metrics.Mean()

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

  if model in ['np_maf', 'sandwich', 'rqs_maf']:
    if model == 'np_maf':
      for i in range(len(new_maf.distribution.bijector.bijectors)):
        if 'batch_normalization' in new_maf.distribution.bijector.bijectors[i].name:
          new_maf.distribution.bijector.bijectors[i].batchnorm.trainable = False
    else:
      for i in range(len(new_maf.bijector.bijectors)):
        if 'batch_normalization' in new_maf.bijector.bijectors[i].name == 'batch_normalization':
          new_maf.bijector.bijectors[i].batchnorm.trainable = False

  plot_heatmap_2d(new_maf, matching_bijector=prior_matching_bijector,
                  mesh_count=500,
                  name=f'{save_dir}/density_{name}.png')
  plt.close()

  eval_dataset = tf.data.Dataset.from_generator(
    functools.partial(generate_2d_data, data=data, batch_size=int(1e6)),
    output_types=tf.float32)

  eval_log_prob = -tf.reduce_mean(new_maf.log_prob(next(iter(eval_dataset))))

  results = {
    'loss': train_loss_results,
    'loss_eval': eval_log_prob,
  }

  if model == 'sandwich':
    for v in new_maf.trainable_variables:
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

    results['samples'] = sample(new_maf, fixed_maf, int(1e3))
  with open(f'{save_dir}/{name}.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print(f'{name} done!')

datasets = ["8gaussians", "2spirals", 'checkerboard', "diamond"]
models = ['sandwich', 'np_maf', 'maf']

main_dir = '2d_toy_results'
if not os.path.isdir(main_dir):
  os.makedirs(main_dir)
n_runs = 5

for run in range(n_runs):
  for data in datasets:
    if not os.path.exists(f'{main_dir}/run_{run}/{data}'):
      os.makedirs(f'{main_dir}/run_{run}/{data}')
    for model in models:
      if model == 'maf':
        name = 'maf'
        train(model, 20, name, save_dir=f'{main_dir}/run_{run}/{data}')
      elif model == 'rqs_maf':
        name = 'rqs_maf'
        for nbins in [8, 128]:
          train(model, 20, name, save_dir=f'{main_dir}/run_{run}/{data}')
      else:
        for n_components in [100]:
          name = f'c{n_components}_{model}'
          train(model, n_components, name, save_dir=f'{main_dir}/run_{run}/{data}')