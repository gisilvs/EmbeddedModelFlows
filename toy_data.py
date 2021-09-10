import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

@tf.function
def eight_gaussians(centers, batch_size):
  point = tf.random.normal([batch_size, 2]) * 0.5
  idxs = tf.squeeze(tf.random.categorical(tf.ones([1, 8]) / 8, batch_size))
  center = tf.gather(centers, idxs)
  point = (point + center) / 1.414
  return point

@tf.function
def checkerboard(batch_size):
  x1 = tf.random.uniform([batch_size]) * 4 - 2
  x2 = tf.random.uniform([batch_size]) - tf.cast(
    tf.squeeze(tf.random.categorical(tf.ones([1, 2]) / 2, batch_size)),
    tf.float32) * 2.
  x2 = x2 + (tf.math.floor(x1) % 2)
  return tf.concat([x1[:, None], x2[:, None]], 1) * 2

@tf.function
def two_spirals(batch_size):
  n = tf.math.sqrt(tf.random.uniform([batch_size // 2, 1])) * 540 * (
        2 * np.pi) / 360
  d1x = -tf.math.cos(n) * n + tf.random.uniform([batch_size // 2, 1]) * 0.5
  d1y = tf.math.sin(n) * n + tf.random.uniform([batch_size // 2, 1]) * 0.5
  x = tf.concat(
    [tf.concat([d1x, d1y], axis=1), tf.concat([-d1x, -d1y], axis=1)],
    axis=0) / 3
  return x + tf.random.normal(tf.shape(x)) * 0.1

@tf.function
def diamond(batch_size, bound, width, covariance_factor, rotation_matrix):
  x = tf.linspace(-bound, bound, width)
  x, y = x[:, None], x[:, None]
  x1 = tf.concat([x, tf.ones_like(x)], axis=-1)
  y1 = tf.concat([tf.ones_like(y), y], axis=-1)
  means = tf.reshape(x1[:, None] * y1[None], (-1, 2))
  means = means + tf.random.uniform(tf.shape(means)) * 1e-3
  index = tf.squeeze(
    tf.random.categorical(tf.ones([1, width ** 2]) / width ** 2, batch_size))
  noise = tf.random.normal([batch_size, 2])
  data = tf.gather(means, index) + noise @ covariance_factor
  return data @ rotation_matrix

# Dataset iterator for generation of dataset samples
def generate_2d_data(data, rng=None, batch_size=1000):
  if rng is None:
    rng = np.random.RandomState()

  if data == "8gaussians":
    scale = 4.
    centers = tf.convert_to_tensor([(1, 0), (-1, 0), (0, 1), (0, -1),
               (1. / np.sqrt(2), 1. / np.sqrt(2)),
               (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                     1. / np.sqrt(2)),
               (-1. / np.sqrt(2), -1. / np.sqrt(2))]) * scale

    while True:
      yield eight_gaussians(centers, batch_size)

  elif data == "diamond":
    bound = -2.5
    width = 15
    covariance_factor = 0.06 * tf.eye(2)
    rotation_matrix = tf.convert_to_tensor(
      [[1 / np.sqrt(2), -1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]],
      dtype=tf.float32)
    while True:
      yield diamond(batch_size, bound, width, covariance_factor, rotation_matrix)

  elif data == "2spirals":
    while True:

      n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
      d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
      d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
      x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
      x += np.random.randn(*x.shape) * 0.1
      yield two_spirals(batch_size)

  elif data == "checkerboard":
    while True:
      yield checkerboard(batch_size)


# distribution generator for initial distribution and sampling
def generate_2d_dist(distribution, rng=None):
  if rng is None:
    rng = np.random.RandomState()

  if distribution == "normal":
    dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2], tf.float32))

    return dist

  elif distribution == "4gaussians":
    mix = [0.25, 0.25, 0.25, 0.25]
    scale = 4.0 / 1.414
    scale_diag = [0.5 / 1.414, 0.5 / 1.414]
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    centers = tf.cast([(scale * x, scale * y) for x, y in centers],
                      dtype=tf.float32)

    dist = tfd.Mixture(
      cat=tfd.Categorical(probs=mix),
      components=[
        tfd.MultivariateNormalDiag(
          loc=centers[0],
          scale_diag=scale_diag),
        tfd.MultivariateNormalDiag(
          loc=centers[1],
          scale_diag=scale_diag),
        tfd.MultivariateNormalDiag(
          loc=centers[2],
          scale_diag=scale_diag),
        tfd.MultivariateNormalDiag(
          loc=centers[3],
          scale_diag=scale_diag)
      ])

    return dist

  elif distribution == "8gaussians":
    mix = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    scale = 4.0 / 1.414
    scale_diag = [0.5 / 1.414, 0.5 / 1.414]
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
               (1. / np.sqrt(2), 1. / np.sqrt(2)),
               (1. / np.sqrt(2), -1. / np.sqrt(2)),
               (-1. / np.sqrt(2), 1. / np.sqrt(2)),
               (-1. / np.sqrt(2), -1. / np.sqrt(2))]

    centers = tf.cast([(scale * x, scale * y) for x, y in centers],
                      dtype=tf.float32)

    dist = tfd.Mixture(
      cat=tfd.Categorical(probs=mix),
      components=[
        tfd.MultivariateNormalDiag(
          loc=centers[0],
          scale_diag=scale_diag),
        tfd.MultivariateNormalDiag(
          loc=centers[1],
          scale_diag=scale_diag),
        tfd.MultivariateNormalDiag(
          loc=centers[2],
          scale_diag=scale_diag),
        tfd.MultivariateNormalDiag(
          loc=centers[3],
          scale_diag=scale_diag),
        tfd.MultivariateNormalDiag(
          loc=centers[4],
          scale_diag=scale_diag),
        tfd.MultivariateNormalDiag(
          loc=centers[5],
          scale_diag=scale_diag),
        tfd.MultivariateNormalDiag(
          loc=centers[6],
          scale_diag=scale_diag),
        tfd.MultivariateNormalDiag(
          loc=centers[7],
          scale_diag=scale_diag)
      ])

    return dist

  else:
    return generate_2d_dist("normal", rng)