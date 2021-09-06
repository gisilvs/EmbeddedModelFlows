import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import cache_util

tfb = tfp.bijectors
tfd = tfp.distributions
tfe = tfp.experimental

class MixtureOfGaussians(tfb.Bijector):

  _cache = cache_util.BijectorCacheWithGreedyAttrs(
    forward_name='_augmented_forward', inverse_name='_augmented_inverse')

  def __init__(self, dist, validate_args=False,
               name='mixture_of_gaussians'):
    # todo: assert dist is mixture of gaussians
    self.dist = dist
    super(MixtureOfGaussians, self).__init__(
      validate_args=validate_args,
      forward_min_event_ndims=0,
      name=name)

  def forward_pass(self, x):
    x = self.dist.cdf(x)
    return x

  def _augmented_forward(self, x):
    fldj = self.dist.log_prob(x)
    return self.forward_pass(x), {'ildj': -fldj, 'fldj': fldj}

  def _augmented_inverse(self, y):
    bij = tfe.bijectors.ScalarFunctionWithInferredInverse(
      lambda e: self.forward_pass(e), max_iterations=50, root_search_fn=tfp.math.find_root_chandrupatla)
    ildj = bij.inverse_log_det_jacobian(y)
    y = bij.inverse(y)
    return y, {'ildj': ildj, 'fldj': -ildj}

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
