import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root


@tfd.JointDistributionCoroutine
def lorenz_system():
  truth = []
  innovation_noise = .1
  step_size = 0.02
  loc = yield Root(tfd.Sample(tfd.Normal(0., 1., name='x_0'), sample_shape=3))
  for t in range(1, 30):
    x, y, z = tf.unstack(loc, axis=-1)
    truth.append(x)
    dx = 10 * (y - x)
    dy = x * (28 - z) - y
    dz = x * y - 8 / 3 * z
    delta = tf.stack([dx, dy, dz], axis=-1)
    loc = yield tfd.Independent(
      tfd.Normal(loc + step_size * delta,
                 tf.sqrt(step_size) * innovation_noise, name=f'x_{t}'),
      reinterpreted_batch_ndims=1)


@tfd.JointDistributionCoroutine
def brownian_motion():
  new = yield Root(tfd.Normal(loc=0, scale=.1))

  for t in range(1, 30):
    new = yield tfd.Normal(loc=new, scale=.1)


@tfd.JointDistributionCoroutine
def ornstein_uhlenbeck():
  a = 0.8
  new = yield Root(tfd.Normal(loc=0, scale=5.))

  for t in range(1, 30):
    new = yield tfd.Normal(loc=a * new, scale=.5)


@tfd.JointDistributionCoroutine
def van_der_pol():
  mul = 4
  innovation_noise = .1
  mu = 1.
  step_size = 0.05
  loc = yield Root(tfd.Sample(tfd.Normal(0., 1., name='x_0'), sample_shape=2))
  for t in range(1, 30 * mul):
    x, y = tf.unstack(loc, axis=-1)
    dx = y
    dy = mu * (1 - x ** 2) * y - x
    delta = tf.stack([dx, dy], axis=-1)
    loc = yield tfd.Independent(
      tfd.Normal(loc + step_size * delta,
                 tf.sqrt(step_size) * innovation_noise, name=f'x_{t}'),
      reinterpreted_batch_ndims=1)


def time_series_gen(batch_size, dataset_name):
  if dataset_name == 'lorenz':
    while True:
      yield tf.reshape(
        tf.transpose(tf.convert_to_tensor(lorenz_system.sample(batch_size)),
                     [1, 0, 2]), [batch_size, -1])
  if dataset_name == 'lorenz_scaled':
    while True:
      samples = tf.convert_to_tensor(lorenz_system.sample(batch_size))
      std = tf.math.reduce_std(samples, axis=1)
      samples = samples / tf.expand_dims(std, 1)
      yield tf.reshape(tf.transpose(samples, [1, 0, 2]), [batch_size, -1])
  if dataset_name == 'van_der_pol':
    while True:
      yield tf.reshape(
        tf.transpose(tf.convert_to_tensor(van_der_pol.sample(batch_size)),
                     [1, 0, 2]), [batch_size, -1])
  elif dataset_name == 'brownian':
    while True:
      yield tf.math.exp(tf.reshape(
        tf.transpose(tf.convert_to_tensor(brownian_motion.sample(batch_size)),
                     [1, 0]), [batch_size, -1]))
  elif dataset_name == 'ornstein':
    while True:
      yield tf.reshape(tf.transpose(
        tf.convert_to_tensor(ornstein_uhlenbeck.sample(batch_size)), [1, 0]),
        [batch_size, -1])
