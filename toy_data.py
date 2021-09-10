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
    rotate=True
    while True:
      means = np.array(
        [
          (x + 1e-3 * np.random.rand(), y + 1e-3 * np.random.rand())
          for x in np.linspace(-bound, bound, width)
          for y in np.linspace(-bound, bound, width)
        ]
      )

      covariance_factor = 0.06 * np.eye(2)

      index = np.random.choice(range(width ** 2), size=batch_size, replace=True)
      noise = np.random.randn(batch_size, 2)
      data = means[index] + noise @ covariance_factor
      if rotate:
        rotation_matrix = np.array(
          [[1 / np.sqrt(2), -1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]]
        )
        data = data @ rotation_matrix
      data = data.astype(np.float32)
      yield data

  elif data == "2spirals":
    while True:
      n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
      d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
      d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
      x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
      x += np.random.randn(*x.shape) * 0.1
      yield np.array(x, dtype='float32')

  elif data == "checkerboard":
    while True:
      x1 = np.random.rand(batch_size) * 4 - 2
      x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
      x2 = x2_ + (np.floor(x1) % 2)
      data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
      yield np.array(data, dtype="float32")


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

'''gen = generate_2d_data('8gaussians')

next(gen)'''