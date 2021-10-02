import os
import shutil
import pickle
import functools
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn import datasets
import surrogate_posteriors

import numpy as np
import matplotlib.pyplot as plt

def iris_generator():
  iris = datasets.load_iris()
  data = iris.data
  labels = iris.target
  class_0 = data[np.where(labels==0)]
  class_1 = data[np.where(labels==1)]
  class_2 = data[np.where(labels==2)]
  while True:
    class_idx = np.random.randint(3)
    if class_idx == 0:
      np.random.shuffle(class_0)
      sample = class_0[:10]
    elif class_idx == 1:
      np.random.shuffle(class_1)
      sample = class_1[:10]
    elif class_idx == 2:
      np.random.shuffle(class_2)
      sample = class_2[:10]
    sample_mean = np.mean(sample, axis=0).reshape(1,-1)
    sample = np.append(sample_mean, sample, axis=0)
    sample = tf.reshape(tf.convert_to_tensor(sample[:10], dtype=tf.float32), [-1])
    yield sample

def digits_generator():
  lambd = 1e-6
  digits = datasets.load_digits()
  data = tf.convert_to_tensor(digits.data, dtype=tf.float32)
  data = (data + tf.random.uniform(tf.shape(data), minval=0., maxval=1., seed=42)) / 17.
  data = tf.math.log(lambd + (1 - 2 * lambd) * data)
  labels = digits.target
  class_dict = {}
  for label in range(10):
    class_dict[label] = tf.gather(data, list(np.where(labels==label)[0]))
  while True:
    class_idx = np.random.randint(10)
    class_dict[class_idx] = tf.random.shuffle(class_dict[class_idx])
    sample = class_dict[class_idx][:20]
    sample_mean = tf.reshape(tf.reduce_mean(sample, axis=0), [1,-1])
    sample = tf.concat([sample_mean, sample], axis=0)
    sample = tf.reshape(sample[:20], [-1])
    yield sample

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root

num_iterations = int(2e5)

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

def train(model, name, dataset_name, save_dir):

  @tf.function
  def optimizer_step(net, inputs):
    with tf.GradientTape() as tape:
      loss = -net.log_prob(inputs)
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss

  def build_model(model_name):
    if dataset_name == 'iris':
      scales = tf.ones(4)
      initial_mean = tf.zeros(4)
      length = 10
    elif dataset_name == 'digits':
      scales = tf.ones(64)
      initial_mean = tf.zeros(64)
      length = 20

    @tfd.JointDistributionCoroutine
    def prior_structure():
      mean = yield Root(tfd.Independent(tfd.Normal(loc=initial_mean,
                                                  scale=scales,
                                                  name='prior0'), 1))

      for t in range(1, length):
        new = yield tfd.Independent(tfd.Normal(loc=mean,
                                               scale=scales, name=f'prior{t}'),
                                    1)

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

    maf.log_prob(prior_structure.sample(1))

    return maf, prior_matching_bijector

  maf, prior_matching_bijector = build_model(model)


  if dataset_name == 'iris':
    dataset = tf.data.Dataset.from_generator(iris_generator,
                                               output_types=tf.float32).map(prior_matching_bijector).batch(100).prefetch(tf.data.AUTOTUNE)
  else:
    dataset = tf.data.Dataset.from_generator(digits_generator,
                                             output_types=tf.float32).map(
      prior_matching_bijector).batch(100).prefetch(tf.data.AUTOTUNE)

  lr = 1e-4
  lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=lr, decay_steps=5e5)
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
    epoch_loss_avg.update_state(loss_value)

    if it == 0:
      best_loss = epoch_loss_avg.result()
      epoch_loss_avg = tf.keras.metrics.Mean()
      save_path = checkpoint_manager.save()
    elif it % 100 == 0:
      train_loss_results.append(epoch_loss_avg.result())
      #print(train_loss_results[-1])
      epoch_loss_avg = tf.keras.metrics.Mean()
      if tf.math.is_nan(train_loss_results[-1]):
        break
      else:
        save_path = checkpoint_manager.save()
    if it >= num_iterations:
      break
    it += 1

  new_maf, _ = build_model(model)

  new_checkpoint = tf.train.Checkpoint(weights=new_maf.trainable_variables)
  new_checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
  if os.path.isdir(f'{save_dir}/checkpoints/{name}'):
    clear_folder(f'{save_dir}/checkpoints/{name}')
  checkpoint_manager = tf.train.CheckpointManager(new_checkpoint,
                                                  f'{save_dir}/checkpoints/{name}',
                                                  max_to_keep=20)
  save_path = checkpoint_manager.save()

  plt.plot(train_loss_results)
  plt.savefig(f'{save_dir}/loss_{name}.png',
              format="png")
  plt.close()

  if dataset_name == 'iris':
    eval_dataset = tf.data.Dataset.from_generator(iris_generator,
                                               output_types=tf.float32).map(prior_matching_bijector).batch(100000)


  else:
    eval_dataset = tf.data.Dataset.from_generator(digits_generator,
                                                  output_types=tf.float32).map(
      prior_matching_bijector).batch(10000)

  eval_log_prob = -tf.reduce_mean(new_maf.log_prob(next(iter(eval_dataset))))


  if dataset_name=='iris':
    samples = tf.convert_to_tensor(new_maf.sample(1000))

  else:
    samples = tf.convert_to_tensor(new_maf.sample(100))

  results = {'samples' : samples,
             'loss_eval': eval_log_prob,
             'loss': train_loss_results
             }
  with open(f'{save_dir}/{name}.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


  print(f'{name} done!')
models = ['np_maf','sandwich','maf']

main_dir = 'hierarchical_results'
if not os.path.isdir(main_dir):
  os.makedirs(main_dir)

dataset = ['digits']

n_runs = [4]

for run in n_runs:
  for data in dataset:
    if not os.path.exists(f'{main_dir}/run_{run}/{data}'):
      os.makedirs(f'{main_dir}/run_{run}/{data}')
    for model in models:
      if model == 'maf':
        name = 'maf'
        train(model, name, dataset_name=data, save_dir=f'{main_dir}/run_{run}/{data}')
      else:
        name = model
        train(model, name, dataset_name=data, save_dir=f'{main_dir}/run_{run}/{data}')
