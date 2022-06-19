import argparse
import logging
import os
import pickle

import tensorflow as tf
import tensorflow_probability as tfp

from models.model_getter import get_model
from utils.metrics import negative_elbo, forward_kl
from vi.models import get_vi_model

parser = argparse.ArgumentParser(description='VI experiments')
parser.add_argument('--model', type=str,
                    help='mean_field | multivariate_normal | asvi | iaf | embedded_model_flow | gated_emf')
parser.add_argument('--vi-model', type=str)
parser.add_argument('--backbone-name', type=str, default='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--run_nr', type=int, default=0)

args = parser.parse_args()

assert args.model in ['mean_field',
                      'multivariate_normal',
                      'asvi',
                      'iaf',
                      'embedded_model_flow',
                      'gated_emf',
                      ]
assert args.vi_models in [
  'van_der_pol_smoothing_r',
  'van_der_pol_smoothing_c',
  'van_der_pol_bridge_r',
  'van_der_pol_bridge_c'
  'eight_schools',
  'radon',
  'brownian_smoothing_r',
  'brownian_smoothing_c',
  'brownian_bridge_r',
  'brownian_bridge_c',
  'lorenz_smoothing_r',
  'lorenz_smoothing_c',
  'lorenz_bridge_r',
  'lorenz_bridge_c',
  'linear_binary_tree_4',
  'linear_binary_tree_8',
  'tanh_binary_tree_4',
  'tanh_binary_tree_8',
]

assert args.backbone_name in ['mean_field',
                              'multivariate_normal',
                              'asvi',
                              'iaf',
                              'embedded_model_flow',
                              'gated_emf',
                              ]

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def train_and_save_results(model_name, surrogate_posterior_name, backbone_name,
                           surrogate_posterior, target_log_prob,
                           ground_truth, observations, learning_rate, i, seed):
  losses = tfp.vi.fit_surrogate_posterior(target_log_prob,
                                          surrogate_posterior,
                                          optimizer=tf.optimizers.Adam(
                                            learning_rate=learning_rate),
                                          num_steps=100000,
                                          sample_size=50)

  if backbone_name:
    surrogate_posterior_name = f'{surrogate_posterior_name}_{backbone_name}'

  repo_name = f'results/{model_name}/{surrogate_posterior_name}'
  checkpoint = tf.train.Checkpoint(
    weights=surrogate_posterior.trainable_variables)
  ckpt_dir = f'{repo_name}/checkpoints_{i}'
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, ckpt_dir,
                                                  max_to_keep=20)
  save_path = checkpoint_manager.save()

  samples = surrogate_posterior.sample(150)

  elbo = negative_elbo(target_log_prob, surrogate_posterior, num_samples=150,
                       seed=seed)

  if ground_truth is not None:
    fkl = forward_kl(surrogate_posterior, ground_truth)
  else:
    fkl = None

  results = {
    'loss': losses,
    'elbo': elbo,
    'fkl': fkl
  }

  if model_name in ['brownian', 'lorenz', 'van_der_pol']:
    results['observations'] = tf.convert_to_tensor(observations)
    results['ground_truth'] = tf.convert_to_tensor(ground_truth)
    results['samples'] = tf.convert_to_tensor(samples)

  if not os.path.exists(repo_name):
    os.makedirs(repo_name)

  with open(f'{repo_name}/rep{i}.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

  print(f'{model_name} {surrogate_posterior_name} rep{i} done!')


if not os.path.exists('all_results/results'):
  os.makedirs('all_results/results')


def main():
  prior, ground_truth, target_log_prob, observations = get_vi_model(
    args.vi_model, seed=args.seed)

  if args.backbone_name == '' or args.model in ['embedded_model_flow',
                                                'gated_emf']:
    backbone = None
  else:
    backbone = args.backbone_name

    model = get_model(prior,
                      args.model_name,
                      backbone)

    train_and_save_results(model_name=args.vi_model,
                           surrogate_posterior_name=args.model,
                           backbone_name=backbone,
                           surrogate_posterior=model,
                           target_log_prob=target_log_prob,
                           ground_truth=ground_truth, observations=observations,
                           learning_rate=args.lr, i=args.run_nr, seed=args.seed)


if __name__ == '__main__':
  main()
