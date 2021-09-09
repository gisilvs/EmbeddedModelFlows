import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

import surrogate_posteriors

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root

n_components = 100
lambd = 1e-6
n_dims = 784
num_epochs = 100


def build_model(model_name, trainable_mixture=True, component_logits=None, locs=None, scales=None):
  if trainable_mixture:
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
         range(n_dims)], name='component_logits')
      locs = tf.Variable(
        [tf.linspace(-15., 0., n_components) for _ in range(n_dims)], name='locs')
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
def optimizer_step():
  with tf.GradientTape() as tape:
    loss = -maf.log_prob(x)
  grads = tape.gradient(loss, maf.trainable_variables)
  optimizer.apply_gradients(zip(grads, maf.trainable_variables))
  return loss

@tf.function
def eval(model, inputs):
  return -model.log_prob(inputs)

def inverse_logits(x):
  return tf.math.exp(x - lambd / (1 - 2 * lambd))

def _preprocess(sample):
  image = tf.cast(sample['image'], tf.float32)
  image = (image + tf.random.uniform(tf.shape(image), minval=0., maxval=1., seed=42)) / 256. # dequantize and
  image = tf.math.log(lambd + (1 - 2 * lambd) * image) # logit
  image = tf.reshape(image, [-1])
  image = prior_matching_bijector(image)
  return image

maf, prior_matching_bijector = build_model('np_maf')

data = tfds.load("mnist", split=["train[:50]", "train[50000:]", "test"])
train_data, valid_data, test_data = data[0], data[1], data[2]

train_dataset = (train_data
                 .map(_preprocess)
                 .batch(256)
                 .prefetch(tf.data.AUTOTUNE)
                 .shuffle(int(10e3)))

valid_dataset = (valid_data
                 .map(_preprocess)
                 .batch(256)
                 .prefetch(tf.data.AUTOTUNE))

test_dataset = (test_data
                 .map(_preprocess)
                 .batch(256)
                 .prefetch(tf.data.AUTOTUNE))


optimizer = tf.optimizers.Adam(learning_rate=1e-4)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, weights=maf.trainable_variables)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, '/tmp/tf_ckpts', max_to_keep=20)
train_loss_results = []
best_val_loss = None
epochs_counter = 0

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  valid_loss_avg = tf.keras.metrics.Mean()
  test_loss_avg = tf.keras.metrics.Mean()
  for x in train_dataset:
    # Optimize the model
    loss_value = optimizer_step()
    print(loss_value)
    epoch_loss_avg.update_state(loss_value)
  break

save_path = checkpoint_manager.save()
print("Saved checkpoint for step {}: {}".format(epoch, save_path))
print()
'''component_logits = tf.convert_to_tensor(component_logits)
locs = tf.convert_to_tensor(locs)
scales = tf.convert_to_tensor(scales)
maf.sample(1)
component_logits = tf.Variable(component_logits)
locs = tf.Variable(locs)
scales = tfp.util.TransformedVariable(scales, tfb.Softplus())'''
train_loss_results.append(epoch_loss_avg.result())

new_maf, _ = build_model('np_maf')
new_optimizer = tf.optimizers.Adam(learning_rate=1e-4)

new_checkpoint = tf.train.Checkpoint(optimizer=new_optimizer, weights=new_maf.trainable_variables)
new_checkpoint.restore(tf.train.latest_checkpoint('/tmp/tf_ckpts'))
new_maf.trainable_variables