import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import dtype_util

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tf.keras
tfkl = tfk.layers

class ActivationNormalization(tfb.Bijector):
  """Bijector to implement Activation Normalization (ActNorm)."""

  def __init__(self, nchan, is_image=True, dtype=tf.float32,
               validate_args=False, name=None):
    parameters = dict(locals())

    self.is_image=is_image
    self._initialized = tf.Variable(False, trainable=False)
    self._m = tf.Variable(tf.zeros(nchan, dtype))
    self._s = tfp.util.TransformedVariable(tf.ones(nchan, dtype),
                                           tfb.Softplus())
    self._bijector = tfb.Invert(
        tfb.Chain([
            tfb.Scale(self._s),
            tfb.Shift(self._m),
        ]))
    super(ActivationNormalization, self).__init__(
        validate_args=validate_args,
        forward_min_event_ndims=1,
        parameters=parameters,
        name=name or 'ActivationNormalization')

  def _inverse(self, y, **kwargs):
    with tf.control_dependencies([self._maybe_init(y, inverse=True)]):
      return self._bijector.inverse(y, **kwargs)

  def _forward(self, x, **kwargs):
    with tf.control_dependencies([self._maybe_init(x, inverse=False)]):
      return self._bijector.forward(x, **kwargs)

  def _inverse_log_det_jacobian(self, y, **kwargs):
    with tf.control_dependencies([self._maybe_init(y, inverse=True)]):
      return self._bijector.inverse_log_det_jacobian(y, 1, **kwargs)

  def _forward_log_det_jacobian(self, x, **kwargs):
    with tf.control_dependencies([self._maybe_init(x, inverse=False)]):
      return self._bijector.forward_log_det_jacobian(x, 1, **kwargs)

  def _maybe_init(self, inputs, inverse):
    """Initialize if not already initialized."""
    is_image = self.is_image
    def _init():
      """Build the data-dependent initialization."""
      if is_image:
        axis = ps.range(ps.rank(inputs) - 1)
        m = tf.math.reduce_mean(inputs, axis=axis)
        s = (
            tf.math.reduce_std(inputs, axis=axis) +
            10. * np.finfo(dtype_util.as_numpy_dtype(inputs.dtype)).eps)
      else:
        axis = ps.range(ps.rank(inputs))
        m = tf.reshape(tf.math.reduce_mean(inputs, axis=axis), [1])
        s = tf.reshape((
            tf.math.reduce_std(inputs, axis=axis) +
            10. * np.finfo(dtype_util.as_numpy_dtype(inputs.dtype)).eps), [1])

      if inverse:
        s = 1 / s
        m = -m
      else:
        m = m / s
      with tf.control_dependencies([self._m.assign(m), self._s.assign(s)]):
        return self._initialized.assign(True)

    return tf.cond(self._initialized, tf.no_op, _init)