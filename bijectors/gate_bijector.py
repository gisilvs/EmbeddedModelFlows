import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import cache_util

tfb = tfp.bijectors
tfd = tfp.distributions
tfe = tfp.experimental


class GateBijectorForNormal(tfb.Bijector):

  def __init__(self, loc, scale, residual_fraction, validate_args=False,
               name='gate_bijector_for_normal'):
    self.loc = loc
    self.scale = scale
    self.residual_fraction = residual_fraction
    super(GateBijectorForNormal, self).__init__(
      validate_args=validate_args,
      forward_min_event_ndims=0,
      name=name)

  def _forward(self, x):
    x = self.residual_fraction * (self.loc + self.scale * x) + \
        (1 - self.residual_fraction) * x
    return x

  def _inverse(self, y):
    y = (y - self.residual_fraction * self.loc) / (
        self.residual_fraction * self.scale + 1 - self.residual_fraction)
    return y

  def _forward_log_det_jacobian(self, x):
    fldj = tf.math.log(
      self.residual_fraction * (self.scale - 1) + 1)
    return fldj


class GateBijector(tfb.Bijector):
  _cache = cache_util.BijectorCacheWithGreedyAttrs(
    forward_name='_augmented_forward', inverse_name='_augmented_inverse')

  def __init__(self, dist_bijector, residual_fraction, validate_args=False,
               name='gate_bijector'):
    self.dist_bijector = dist_bijector
    self.residual_fraction = residual_fraction
    super(GateBijector, self).__init__(
      validate_args=validate_args,
      forward_min_event_ndims=0,
      name=name)

  def _gating_bijector(self, gated_residual_fraction):
    return (tfe.bijectors.ScalarFunctionWithInferredInverse(
      lambda x, lam: lam * self.dist_bijector(x) + (1 - lam) * x,
      additional_scalar_parameters_requiring_gradients=[
        gated_residual_fraction]))

  def _augmented_forward(self, x):
    bij = self._gating_bijector(tf.convert_to_tensor(self.residual_fraction))
    fldj = bij.forward_log_det_jacobian(x)
    return bij.forward(x), {'ildj': -fldj, 'fldj': fldj}

  def _augmented_inverse(self, y):
    bij = self._gating_bijector(tf.convert_to_tensor(self.residual_fraction))
    ildj = bij.inverse_log_det_jacobian(y)
    return bij.inverse(y), {'ildj': ildj, 'fldj': -ildj}

  def _forward(self, x):
    y, _ = self._augmented_forward(x)
    return y

  def _inverse(self, y):
    x, _ = self._augmented_inverse(y)
    return x

  def _forward_log_det_jacobian(self, x):
    cached = self._cache.forward_attributes(x)
    # If LDJ isn't in the cache, call forward once.
    if 'fldj' not in cached:
      _, attrs = self._augmented_forward(x)
      cached.update(attrs)
    return cached['fldj']

  def _inverse_log_det_jacobian(self, y):
    cached = self._cache.inverse_attributes(y)
    # If LDJ isn't in the cache, call inverse once.
    if 'ildj' not in cached:
      _, attrs = self._augmented_inverse(y)
      cached.update(attrs)
    return cached['ildj']
