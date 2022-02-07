import os
import logging
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from metrics import negative_elbo, forward_kl
from models import get_model
import surrogate_posteriors
from surrogate_posteriors import get_surrogate_posterior

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
logging.getLogger('tensorflow').setLevel(logging.ERROR)

learning_rates = {'mean_field': 1e-3,
                  'multivariate_normal': 1e-3,
                  'asvi': 1e-3,
                  'iaf': 5e-5,
                  'highway_flow': 1e-4}


def train_and_save_results(model_name, surrogate_posterior_name, backbone_name, surrogate_posterior, target_log_prob,
                           ground_truth, observations, learning_rate, i, seed):



  losses = tfp.vi.fit_surrogate_posterior(target_log_prob,
                                          surrogate_posterior,
                                          optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                                          num_steps=100000,
                                          sample_size=50)

  if backbone_name:
    surrogate_posterior_name = f'{surrogate_posterior_name}_{backbone_name}'

  repo_name = f'results/{model_name}/{surrogate_posterior_name}'
  checkpoint = tf.train.Checkpoint(weights=surrogate_posterior.trainable_variables)
  ckpt_dir = f'{repo_name}/checkpoints_{i}'
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, ckpt_dir,
                                                  max_to_keep=20)
  save_path = checkpoint_manager.save()

  samples = surrogate_posterior.sample(150)

  elbo = negative_elbo(target_log_prob, surrogate_posterior, num_samples=150, seed=seed)

  if ground_truth is not None:
    fkl = forward_kl(surrogate_posterior, ground_truth)
  else:
    fkl = None

  results = {
    'loss': losses,
    'elbo': elbo,
    'fkl': fkl
  }
  if surrogate_posterior_name == 'gated_normalizing_program':
    results['residual_fraction_vars'] = surrogate_posteriors.residual_fraction_vars

  if 'brownian' in model_name or 'lorenz' in model_name or 'van_der_pol' in \
      model_name:
    results['observations'] = tf.convert_to_tensor(observations)
    results['ground_truth'] = tf.convert_to_tensor(ground_truth)
    results['samples'] = tf.convert_to_tensor(samples)


  if not os.path.exists(repo_name):
    os.makedirs(repo_name)

  with open(f'{repo_name}/rep{i}.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

  print(f'{model_name} {surrogate_posterior_name} rep{i} done!')

if not os.path.exists('results'):
  os.makedirs('results')

#todo: test more radon
model_names = [
    'van_der_pol_smoothing_r',
    'van_der_pol_smoothing_c',
    'van_der_pol_bridge_r',
    'van_der_pol_bridge_c'
    #'eight_schools',
               #'radon',
               #'brownian_smoothing_r',
               #'brownian_smoothing_c',
               #'brownian_bridge_r',
               #'brownian_bridge_c',
               #'lorenz_smoothing_r',
               #'lorenz_smoothing_c',
               #'lorenz_bridge_r',
               #'lorenz_bridge_c',
               #'linear_binary_tree_4',
               #'linear_binary_tree_8',
               #'tanh_binary_tree_4',
               #'tanh_binary_tree_8',
               ]

surrogate_posterior_names = [# 'mean_field',
                             #'multivariate_normal',
                             #'asvi',
                             #'iaf',
                             'normalizing_program',
                             'gated_normalizing_program'
]

backbone_names = ['mean_field',
                  #'multivariate_normal',
                  #'iaf',
                  #'highway_flow'
]


seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#seeds = [10]
#seeds = [10, 30, 50, 70, 90]

for i in range(10):
  for model_name in model_names:
    prior, ground_truth, target_log_prob, observations = get_model(
      model_name, seed=seeds[i])
    for surrogate_posterior_name in surrogate_posterior_names:
      if surrogate_posterior_name == 'normalizing_program' or surrogate_posterior_name=='gated_normalizing_program':
        for backbone_name in backbone_names:
          surrogate_posterior = get_surrogate_posterior(prior,
                                                        surrogate_posterior_name,
                                                        backbone_name)
          train_and_save_results(model_name=model_name,
                                 surrogate_posterior_name=surrogate_posterior_name,
                                 backbone_name=backbone_name,
                                 surrogate_posterior=surrogate_posterior,
                                 target_log_prob=target_log_prob,
                                 ground_truth=ground_truth, observations=observations,
                                 learning_rate=learning_rates[backbone_name], i=i, seed=seeds[i])

      else:
        surrogate_posterior = get_surrogate_posterior(prior,
                                                      surrogate_posterior_name,
                                                      None)
        train_and_save_results(model_name=model_name,
                               surrogate_posterior_name=surrogate_posterior_name,
                               backbone_name=None,
                               surrogate_posterior=surrogate_posterior,
                               target_log_prob=target_log_prob,
                               ground_truth=ground_truth, observations=observations,
                               learning_rate=learning_rates[surrogate_posterior_name], i=i, seed=seeds[i])

# todo: how do I save a fitted surrogate posterior (as if it was a neural network?)
