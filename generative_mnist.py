import argparse
import os
import pickle

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from utils.generative_utils import get_mixture_prior, build_model
from utils.utils import clear_folder

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root

parser = argparse.ArgumentParser(description='Generative experiments')
parser.add_argument('--model', type=str,
                    help='maf | maf3| emf_t | emf_m | splines | nsf_emf_t | nsf_emf_m')
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--backbone-name', type=str, default='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--run_nr', type=int, default=0)
parser.add_argument('--main_dir', type=str, default='all_results/new_results')

args = parser.parse_args()


def main():
  lambd = 1e-6

  def _preprocess(sample):
    image = tf.cast(sample['image'], tf.float32)
    image = (image + tf.random.uniform(tf.shape(image), minval=0., maxval=1.,
                                       seed=42)) / 256.  # dequantize
    image = lambd + (1 - 2 * lambd) * image
    image = tfb.Invert(tfb.Sigmoid())(image)  # logit
    image = tf.reshape(image, [-1])
    image = prior_matching_bijector(image)
    return image

  @tf.function
  def optimizer_step(net, inputs):
    with tf.GradientTape() as tape:
      loss = -net.log_prob(inputs)
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss

  assert args.model in ['maf', 'maf3', 'emf_t', 'emf_m', 'gemf_t', 'gemf_m',
                        'splines', 'nsf_emf_t', 'nsf_emf_m', 'maf_b']

  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

  num_iterations = args.num_iterations

  main_dir = args.main_dir
  if not os.path.isdir(main_dir):
    os.makedirs(main_dir)

  run = args.run_nr
  flow_dim = 784

  if not os.path.exists(f'{main_dir}/run_{run}/{args.dataset}'):
    os.makedirs(f'{main_dir}/run_{run}/{args.dataset}')

  save_dir = f'{main_dir}/run_{run}/{args.dataset}'

  if 'emf' in args.model:
    trainable_mixture = True
  else:
    trainable_mixture = False
  prior_structure = get_mixture_prior(args.model,
                                      n_components=100,
                                      n_dims=flow_dim,
                                      trainable_mixture=trainable_mixture)

  model, prior_matching_bijector = build_model(args.model, prior_structure,
                                               flow_dim=flow_dim)

  data = tfds.load("mnist", split=["train[:50000]", "train[50000:]", "test"])
  train_data, valid_data, test_data = data[0], data[1], data[2]

  train_dataset = (train_data
                   .map(map_func=_preprocess,
                        num_parallel_calls=tf.data.AUTOTUNE)
                   .cache()
                   .shuffle(int(1e3))
                   .batch(256)
                   .prefetch(tf.data.AUTOTUNE))

  valid_dataset = (valid_data
                   .map(map_func=_preprocess,
                        num_parallel_calls=tf.data.AUTOTUNE)
                   .cache()
                   .batch(256)
                   .prefetch(tf.data.AUTOTUNE))

  test_dataset = (test_data
                  .map(map_func=_preprocess,
                       num_parallel_calls=tf.data.AUTOTUNE)
                  .cache()
                  .batch(256)
                  .prefetch(tf.data.AUTOTUNE))

  lr = args.lr
  optimizer = tf.optimizers.Adam(learning_rate=lr)
  train_loss_results = []
  valid_loss_results = []
  best_val_loss = None
  epochs_counter = 0

  for epoch in range(args.num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    valid_loss_avg = tf.keras.metrics.Mean()
    for x in train_dataset:
      # Optimize the model
      loss_value = optimizer_step(model, x)
      # print(loss_value)
      epoch_loss_avg.update_state(loss_value)
    train_loss_results.append(epoch_loss_avg.result())
    if epoch % 100 == 0:
      print(epoch)
      print(train_loss_results[-1])
    for x in valid_dataset:
      loss_value = eval(model, x)
      valid_loss_avg.update_state(loss_value)
    valid_loss_results.append(valid_loss_avg.result())

    if tf.math.is_nan(valid_loss_avg.result()):
      break

    if epoch == 0:
      checkpoint = tf.train.Checkpoint(weights=model.trainable_variables)
      ckpt_dir = f'/tmp/{save_dir}/checkpoints/{args.model}'
      if os.path.isdir(ckpt_dir):
        clear_folder(ckpt_dir)
      checkpoint_manager = tf.train.CheckpointManager(checkpoint, ckpt_dir,
                                                      max_to_keep=20)
      best_loss = valid_loss_avg.result()

    elif best_loss > valid_loss_avg.result():
      test_loss_avg = tf.keras.metrics.Mean()
      for x in test_dataset:
        loss_value = eval(model, x)
        test_loss_avg.update_state(loss_value)
      if not tf.math.is_nan(test_loss_avg.result()):
        save_path = checkpoint_manager.save()
      best_loss = valid_loss_avg.result()
      epochs_counter = 0
    else:
      epochs_counter += 1
      if epochs_counter > 100:
        print(f'Stop at epoch {epoch}')
        break

  new_model, _ = build_model(args.model)
  new_model.log_prob(x)
  eval(new_model, x)
  new_checkpoint = tf.train.Checkpoint(weights=new_model.trainable_variables)

  new_checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  if os.path.isdir(f'{save_dir}/checkpoints/{args.model}'):
    clear_folder(f'{save_dir}/checkpoints/{args.model}')
  checkpoint_manager = tf.train.CheckpointManager(new_checkpoint,
                                                  f'{save_dir}/checkpoints/{args.model}',
                                                  max_to_keep=20)
  save_path = checkpoint_manager.save()

  plt.plot(train_loss_results)
  plt.plot(valid_loss_results)
  plt.savefig(f'{save_dir}/loss_{args.model}.png',
              format="png")
  plt.close()

  test_loss_avg = tf.keras.metrics.Mean()

  for x in test_dataset:
    loss_value = eval(new_model, x)
    test_loss_avg.update_state(loss_value)

  results = {  # 'samples': tf.convert_to_tensor(new_maf.sample(1000)),
    'loss_eval': test_loss_avg.result(),
    'train_loss': train_loss_results,
    'valid_loss': valid_loss_results
  }

  with open(f'{save_dir}/{args.model}.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print(f'{args.model} done!')


if __name__ == '__main__':
  main()
