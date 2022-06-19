import argparse
import functools
import os
import pickle

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from datasets.toy_data_2d_generator import generate_2d_data
from utils.generative_utils import get_mixture_prior, build_model, \
  get_timeseries_prior
from utils.plot_utils import plot_heatmap_2d
from utils.utils import clear_folder

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root

parser = argparse.ArgumentParser(description='Generative experiments')
parser.add_argument('--model', type=str,
                    help='maf | maf3| emf_t | emf_m | splines | nsf_emf_t | nsf_emf_m')
parser.add_argument('--prior', type=str,
                    help='mixture | continuity | smoothness | hierarchical')
parser.add_argument('--dataset', type=str)
parser.add_argument('--num_iterations', type=int)
parser.add_argument('--backbone-name', type=str, default='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--run_nr', type=int, default=0)
parser.add_argument('--main_dir', type=str, default='all_results/new_results')

args = parser.parse_args()

def main():
  @tf.function
  def optimizer_step(net, inputs):
    with tf.GradientTape() as tape:
      loss = -net.log_prob(inputs)
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss

  assert args.model in ['maf', 'maf3', 'emf_t', 'emf_m', 'gemf_t', 'gemf_m',
                        'splines', 'nsf_emf_t', 'nsf_emf_m', 'maf_b']
  assert args.dataset in ['8gaussians', 'checkerboard', 'mnist', 'brownian',
                          'ornstein' 'lorenz', 'van_der_pol']

  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

  num_iterations = args.num_iterations

  main_dir = args.main_dir
  if not os.path.isdir(main_dir):
    os.makedirs(main_dir)

  run = args.run_nr
  if args.dataset in ['8gaussians', 'checkerboard']:
    dataset = tf.data.Dataset.from_generator(
      functools.partial(generate_2d_data, data=args.dataset,
                        batch_size=int(100)),
      output_types=tf.float32)
    flow_dim = 2

  if args.dataset in ['brownian', 'ornstein' 'lorenz', 'van_der_pol']:
    if args.dataset == 'lorenz':
      time_step_dim = 3
      series_len = 30

    elif args.dataset in ['brownian', 'ornstein']:
      time_step_dim = 1
      series_len = 30

    elif args.dataset == 'van_der_pol':
      time_step_dim = 2
      series_len = 120

    flow_dim = series_len * time_step_dim

  if not os.path.exists(f'{main_dir}/run_{run}/{args.dataset}'):
    os.makedirs(f'{main_dir}/run_{run}/{args.dataset}')

  save_dir = f'{main_dir}/run_{run}/{args.dataset}'

  if args.prior == 'mixture':
    if 'emf' in args.model:
      trainable_mixture = True
    else:
      trainable_mixture = False
    prior_structure = get_mixture_prior(args.model,
                                        n_components=100,
                                        n_dims=flow_dim,
                                        trainable_mixture=trainable_mixture)

  elif args.prior in ['continuity', 'smoothness']:
    prior_structure = get_timeseries_prior(args.model, args.prior,
                                           time_step_dim)

  model, prior_matching_bijector = build_model(args.model, prior_structure,
                                               flow_dim=flow_dim)

  dataset = dataset.map(prior_matching_bijector).prefetch(tf.data.AUTOTUNE)
  lr = args.lr
  lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=lr, decay_steps=args.num_iterations)
  optimizer = tf.optimizers.Adam(learning_rate=lr_decayed_fn)
  checkpoint = tf.train.Checkpoint(weights=model.trainable_variables)
  ckpt_dir = f'/tmp/{save_dir}/checkpoints/{args.model}'
  if os.path.isdir(ckpt_dir):
    clear_folder(ckpt_dir)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, ckpt_dir,
                                                  max_to_keep=20)
  train_loss_results = []
  epoch_loss_avg = tf.keras.metrics.Mean()
  it = 0
  for x in dataset:

    loss_value = optimizer_step(model, x)
    # print(loss_value)
    epoch_loss_avg.update_state(loss_value)
    if it == 0:
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
    it += 1

  new_model, _ = build_model(args.model, prior_structure, flow_dim)
  new_checkpoint = tf.train.Checkpoint(weights=new_model.trainable_variables)

  new_checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  if os.path.isdir(f'{save_dir}/checkpoints/{args.model}'):
    clear_folder(f'{save_dir}/checkpoints/{args.model}')
  checkpoint_manager = tf.train.CheckpointManager(new_checkpoint,
                                                  f'{save_dir}/checkpoints/{args.model}',
                                                  max_to_keep=20)
  save_path = checkpoint_manager.save()

  plt.plot(train_loss_results)
  plt.savefig(f'{save_dir}/loss_{args.model}.png',
              format="png")
  plt.close()

  plot_heatmap_2d(new_model, matching_bijector=prior_matching_bijector,
                  mesh_count=500,
                  name=f'{save_dir}/density_{args.model}.png')
  plt.close()

  if args.dataset in ['8gaussians', 'checkerboard']:
    eval_dataset = tf.data.Dataset.from_generator(
      functools.partial(generate_2d_data, data=args.dataset,
                        batch_size=int(1e4)),
      output_types=tf.float32).map(prior_matching_bijector)

  eval_log_prob = -tf.reduce_mean(new_model.log_prob(next(iter(eval_dataset))))

  results = {
    'loss': train_loss_results,
    'loss_eval': eval_log_prob,
  }

  with open(f'{save_dir}/{args.model}.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print(f'{args.model} done!')


if __name__ == '__main__':
  main()
