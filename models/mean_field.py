import tensorflow as tf
import tensorflow_probability as tfp

from utils.utils import get_prior_matching_bijectors_and_event_dims

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental
tfp_util = tfp.util


def mean_field(prior):
  event_shape, flat_event_shape, flat_event_size, ndims, dtype, prior_matching_bijectors = get_prior_matching_bijectors_and_event_dims(
    prior)
  base_dist = tfd.Independent(tfd.Normal(loc=tf.Variable(tf.reshape(
    [0. for _ in
     range(int(tf.reduce_sum(flat_event_size)))], -1)),
    scale=tfp.util.TransformedVariable(tf.reshape(
      [1.
       for _ in
       range(int(tf.reduce_sum(flat_event_size)))],
      -1), bijector=tfb.Softplus())), 1)
  return tfd.TransformedDistribution(
    distribution=base_dist,
    bijector=tfb.Chain(prior_matching_bijectors))
