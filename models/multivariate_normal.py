import tensorflow as tf
import tensorflow_probability as tfp

from utils.utils import get_prior_matching_bijectors_and_event_dims

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental
tfp_util = tfp.util


def multivariate_normal(prior):
  def make_trainable_linear_operator_tril(
      dim,
      scale_initializer=1e-1,
      diag_bijector=None,
      diag_shift=1e-5,
      dtype=tf.float32):
    """Build a trainable lower triangular linop."""
    scale_tril_bijector = tfb.FillScaleTriL(
      diag_bijector, diag_shift=diag_shift)
    flat_initial_scale = tf.zeros((dim * (dim + 1) // 2,), dtype=dtype)
    initial_scale_tril = tfb.FillScaleTriL(
      diag_bijector=tfb.Identity(), diag_shift=scale_initializer)(
      flat_initial_scale)
    return tf.linalg.LinearOperatorLowerTriangular(
      tril=tfp_util.TransformedVariable(
        initial_scale_tril, bijector=scale_tril_bijector))

  event_shape, flat_event_shape, flat_event_size, ndims, dtype, prior_matching_bijectors = get_prior_matching_bijectors_and_event_dims(
    prior)
  base_dist = tfd.Sample(
    tfd.Normal(tf.zeros([], dtype), 1.), sample_shape=[ndims])
  op = make_trainable_linear_operator_tril(ndims)

  prior_matching_bijectors.extend(
    [tfb.Shift(tf.Variable(tf.zeros([ndims], dtype=dtype))),
     tfb.ScaleMatvecLinearOperator(op)])

  return tfd.TransformedDistribution(base_dist,
                                     tfb.Chain(prior_matching_bijectors))
