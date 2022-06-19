import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

import bijector_test_util

from bijectors.gate_bijector import GateBijectorForNormal, GateBijector
from bijectors.gated_structured_layer import GatedStructuredLayer

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

  def testTheoreticalFldj(self):
    x = samplers.uniform([10], minval=-1., maxval=1., seed=(0, 0))
    bijector = GateBijectorForNormal(3., 2., tfp.util.TransformedVariable(0.98,
                                                                          bijector=tfb.Sigmoid()))
    self.evaluate([v.initializer for v in bijector.trainable_variables])
    y = bijector.forward(x)
    bijector_test_util.assert_bijective_and_finite(
      bijector,
      x,
      y,
      eval_func=self.evaluate,
      event_ndims=1,
      inverse_event_ndims=1,
      rtol=1e-5)

    fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
    # The jacobian is not yet broadcast, since it is constant.
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
      bijector, x, event_ndims=1)
    self.assertAllClose(
      self.evaluate(fldj_theoretical),
      self.evaluate(fldj),
      atol=1e-5,
      rtol=1e-5)

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

    bijector = GatedStructuredLayer(simple_prior)

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


@test_util.test_all_tf_execution_regimes
class GateBijectorTests(test_util.TestCase):

  def testBijector(self):
    x = samplers.uniform([3], minval=-1., maxval=1., seed=(0,0))

    bijector = GateBijector(tfb.Shift(3.)(tfb.Scale(2.)), tfp.util.TransformedVariable(0.98,
                                                          bijector=tfb.Sigmoid()))

    self.evaluate([v.initializer for v in bijector.trainable_variables])
    self.assertStartsWith(bijector.name, 'gate_bijector')
    self.assertAllClose(x, bijector.inverse(tf.identity(bijector.forward(x))))
    self.assertAllClose(
      bijector.forward_log_det_jacobian(x, event_ndims=1),
      -bijector.inverse_log_det_jacobian(
        tf.identity(bijector.forward(x)), event_ndims=1))

  def testTheoreticalFldj(self):
    x = samplers.uniform([10], minval=-1., maxval=1., seed=(0, 0))
    bijector = GateBijector(tfb.Shift(3.)(tfb.Scale(2.)), tfp.util.TransformedVariable(0.98,
                                                          bijector=tfb.Sigmoid()))
    self.evaluate([v.initializer for v in bijector.trainable_variables])
    y = bijector.forward(x)
    bijector_test_util.assert_bijective_and_finite(
      bijector,
      x,
      y,
      eval_func=self.evaluate,
      event_ndims=1,
      inverse_event_ndims=1,
      rtol=1e-5)

    fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
    # The jacobian is not yet broadcast, since it is constant.
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
      bijector, x, event_ndims=1)
    self.assertAllClose(
      self.evaluate(fldj_theoretical),
      self.evaluate(fldj),
      atol=1e-5,
      rtol=1e-5)