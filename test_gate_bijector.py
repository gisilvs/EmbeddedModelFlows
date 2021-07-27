import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

from surrogate_posteriors import get_surrogate_posterior, GatedAutoFromNormal

from gate_bijector import GateBijectorForNormal

tfb = tfp.bijectors
tfd = tfp.distributions

@test_util.test_all_tf_execution_regimes
class GateBijectorForNormalTests(test_util.TestCase):

  def testBijector(self):
    x = samplers.uniform([3], minval=-1., maxval=1., seed=(0,0))

    bijector = GateBijectorForNormal(3., 2., tfp.util.TransformedVariable(0.98,
                                                          bijector=tfb.Sigmoid()))

    self.evaluate([v.initializer for v in bijector.trainable_variables])
    self.assertStartsWith(bijector.name, 'gate_bijector_for_normal')
    self.assertAllClose(x, bijector.inverse(tf.identity(bijector.forward(x))))
    self.assertAllClose(
      bijector.forward_log_det_jacobian(x, event_ndims=1),
      -bijector.inverse_log_det_jacobian(
        tf.identity(bijector.forward(x)), event_ndims=1))

  def testGradientsSimplecase(self):
    bijector = GateBijectorForNormal(3., 2., tfp.util.TransformedVariable(0.98,
                                                                          bijector=tfb.Sigmoid()))
    dist = tfd.TransformedDistribution(
    distribution=tfd.Sample(tfd.Normal(0., 1.), 3),
    bijector=bijector
  )

    with tf.GradientTape() as tape:
      posterior_sample = dist.sample(
        seed=(0, 0))
      posterior_logprob = dist.log_prob(posterior_sample)
    grad = tape.gradient(posterior_logprob,
                         dist.trainable_variables)

    self.assertTrue(all(g is not None for g in grad))

  def testGradientsWithCoroutine(self):
    @tfd.JointDistributionCoroutineAutoBatched
    def simple_prior():
      yield tfd.Sample(tfd.Normal(0., 1.), 3)

    bijector = GatedAutoFromNormal(simple_prior)

    surrogate_posterior = tfd.TransformedDistribution(
      distribution=simple_prior,
      bijector=bijector
    )

    with tf.GradientTape() as tape:
      posterior_sample, posterior_logprob = surrogate_posterior.experimental_sample_and_log_prob(
        seed=(0, 0))
      #posterior_logprob = surrogate_posterior.log_prob(posterior_sample)
    grad = tape.gradient(posterior_logprob,
                         surrogate_posterior.trainable_variables)

    self.assertTrue(all(g is not None for g in grad))


# TODO: add test for jacobian

