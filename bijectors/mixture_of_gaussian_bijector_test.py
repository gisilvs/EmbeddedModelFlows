import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util

import bijector_test_util

from bijectors.mixture_of_gaussian_bijector import InverseMixtureOfGaussians

tfb = tfp.bijectors
tfd = tfp.distributions
Root = tfd.JointDistributionCoroutine.Root

n_components = 100
n_dims = 2
component_logits = tf.convert_to_tensor(
  [[1. / n_components for _ in range(n_components)] for _ in
   range(n_dims)])
locs = tf.convert_to_tensor([[0. for _ in range(n_components)] for _ in range(n_dims)])
scales = tf.convert_to_tensor(
  [[1. for _ in range(n_components)] for _ in
   range(n_dims)])

dist = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(logits=component_logits),
    components_distribution=tfd.Normal(loc=locs, scale=scales),
    name=f"prior")

@test_util.test_all_tf_execution_regimes
class GateBijectorForNormalTests(test_util.TestCase):

  def testBijector(self):
    x = tfb.NormalCDF()(dist.sample(10, seed=(0,0)))

    bijector = InverseMixtureOfGaussians(dist)

    self.evaluate([v.initializer for v in bijector.trainable_variables])
    self.assertStartsWith(bijector.name, 'mixture_of_gaussians')
    self.assertAllClose(tf.convert_to_tensor(x), bijector.inverse(tf.identity(bijector.forward(x))))

  def testTheoreticalFldj(self):
    x = tfb.NormalCDF()(dist.sample(10, seed=(0,0)))
    bijector = InverseMixtureOfGaussians(dist)
    self.evaluate([v.initializer for v in bijector.trainable_variables])
    y = bijector.forward(x)
    bijector_test_util.assert_bijective_and_finite(
      bijector,
      tf.convert_to_tensor(x),
      y,
      eval_func=self.evaluate,
      event_ndims=1,
      inverse_event_ndims=1,
      rtol=1e-5)

    fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
    # The jacobian is not yet broadcast, since it is constant.
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
      bijector, tf.convert_to_tensor(x), event_ndims=1)
    self.assertAllClose(
      self.evaluate(fldj_theoretical),
      self.evaluate(fldj),
      atol=1e-5,
      rtol=1e-5)
