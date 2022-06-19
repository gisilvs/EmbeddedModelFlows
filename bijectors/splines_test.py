import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

import bijector_test_util

from bijectors.splines import NeuralSplineFlow

tfb = tfp.bijectors
tfd = tfp.distributions
Root = tfd.JointDistributionCoroutine.Root




@test_util.test_all_tf_execution_regimes
class NeuralSplinesTests(test_util.TestCase):

  def testBijector(self):
    input_dim = 10
    x = samplers.uniform([3, input_dim], minval=-1., maxval=1., seed=(0,0))
    bijector = NeuralSplineFlow(input_dim=input_dim, d_dim=int(input_dim / 2) + 1,
                                b_interval=[3 for _ in range(input_dim)])

    self.evaluate([v.initializer for v in bijector.trainable_variables])
    self.assertStartsWith(bijector.name, 'neural_spline_flow')
    self.assertAllClose(tf.convert_to_tensor(x), bijector.inverse(tf.identity(bijector.forward(x))))
    self.assertAllClose(
      bijector.forward_log_det_jacobian(x, event_ndims=1),
      -bijector.inverse_log_det_jacobian(
        tf.identity(bijector.forward(x)), event_ndims=1))

  def testTheoreticalFldj(self):
    input_dim = 2
    x = tf.random.uniform([2, input_dim], minval=-1., maxval=1.)

    bijector = NeuralSplineFlow(input_dim, 2, [3, 3])
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
