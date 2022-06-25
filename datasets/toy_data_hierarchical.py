import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn import datasets

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root


def iris_generator():
  iris = datasets.load_iris()
  data = iris.data
  labels = iris.target
  class_0 = data[np.where(labels == 0)]
  class_1 = data[np.where(labels == 1)]
  class_2 = data[np.where(labels == 2)]
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
    sample_mean = np.mean(sample, axis=0).reshape(1, -1)
    sample = np.append(sample_mean, sample, axis=0)
    sample = tf.reshape(tf.convert_to_tensor(sample[:10], dtype=tf.float32),
                        [-1])
    yield sample


def digits_generator():
  lambd = 1e-6
  digits = datasets.load_digits()
  data = tf.convert_to_tensor(digits.data, dtype=tf.float32)
  data = (data + tf.random.uniform(tf.shape(data), minval=0., maxval=1.,
                                   seed=42)) / 17.
  data = lambd + (1 - 2 * lambd) * data
  data = tfb.Invert(tfb.Sigmoid())(data)  # logit
  labels = digits.target
  class_dict = {}
  for label in range(10):
    class_dict[label] = tf.gather(data, list(np.where(labels == label)[0]))
  while True:
    class_idx = np.random.randint(10)
    class_dict[class_idx] = tf.random.shuffle(class_dict[class_idx])
    sample = class_dict[class_idx][:20]
    sample_mean = tf.reshape(tf.reduce_mean(sample, axis=0), [1, -1])
    sample = tf.concat([sample_mean, sample], axis=0)
    sample = tf.reshape(sample[:20], [-1])
    yield sample
