import os
import pickle
import tensorflow as tf
import tensorflow_probability as tfp

from metrics import negative_elbo, forward_kl
from models import get_model
from surrogate_posteriors import get_surrogate_posterior


def train_and_save_results(model_name, surrogate_posterior_name, backbone_name, surrogate_posterior, target_log_prob,
                           ground_truth, i):
  losses = tfp.vi.fit_surrogate_posterior(target_log_prob,
                                          surrogate_posterior,
                                          optimizer=tf.optimizers.Adam(
                                            learning_rate=1e-3),
                                          num_steps=100000,
                                          sample_size=50)
  elbo = negative_elbo(target_log_prob, surrogate_posterior, num_smaples=15,
                       model_name=model_name)

  if ground_truth is not None:
    fkl = forward_kl(surrogate_posterior, ground_truth)
  else:
    fkl = None

  results = {'loss':losses,
             'elbo':elbo,
             'fkl':fkl}

  if backbone_name:
    surrogate_posterior_name = f'{surrogate_posterior_name}_{backbone_name}'

  repo_name = f'results/{model_name}/{surrogate_posterior_name}'
  if not os.path.exists(repo_name):
    os.makedirs(repo_name)

  with open(f'{repo_name}/rep{i}.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

  print(f'{model_name} {surrogate_posterior_name} rep{i} done!')

if not os.path.exists('results'):
  os.makedirs('results')
model_names = ['brownian_bridge', 'lorenz_bridge', 'eight_schools', 'radon',
               'linear_binary_tree_small', 'linear_binary_tree_large',
               'tanh_binary_tree_small', 'tanh_binary_tree_large']
surrogate_posterior_names = ['mean_field', 'multivariate_normal', 'asvi',
                             'small_iaf', 'large_iaf', 'highway_flow',
                             'normalizing_program']

backbone_names = ['mean_field', 'multivariate_normal', 'large_iaf',
                  'highway_flow']

for i in range(10):
  for model_name in model_names:
    model, prior, ground_truth, target_log_prob, observations = get_model(
      model_name)
    for surrogate_posterior_name in surrogate_posterior_names:
      if surrogate_posterior_name == 'normalizing_program':
        for backbone_name in backbone_names:
          surrogate_posterior = get_surrogate_posterior(prior,
                                                        surrogate_posterior_name,
                                                        backbone_name)
          train_and_save_results(model_name=model_name,
                                 surrogate_posterior_name=surrogate_posterior_name,
                                 backbone_name=backbone_name,
                                 surrogate_posterior=surrogate_posterior,
                                 target_log_prob=target_log_prob,
                                 ground_truth=ground_truth, i=i)

      else:
        surrogate_posterior = get_surrogate_posterior(prior,
                                                      surrogate_posterior_name,
                                                      None)
        train_and_save_results(model_name=model_name,
                               surrogate_posterior_name=surrogate_posterior_name,
                               backbone_name=None,
                               surrogate_posterior=surrogate_posterior,
                               target_log_prob=target_log_prob,
                               ground_truth=ground_truth, i=i)

# todo: how do I save a fitted surrogate posterior (as if it was a neural network?)
