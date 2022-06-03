import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from .actnorm import ActivationNormalization

from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import dtype_util

class SplineParams(tf.Module):

  def __init__(self, nbins=32, interval_width=2, range_min=-1,
               min_bin_width=1e-3, min_slope=1e-3):
    self._nbins = nbins
    self._interval_width = interval_width  # Sum of bin widths.
    self._range_min = range_min  # Position of first knot.
    self._min_bin_width = min_bin_width  # Bin width lower bound.
    self._min_slope = min_slope  # Lower bound for slopes at internal knots.
    self._built = False
    self._bin_widths = None
    self._bin_heights = None
    self._knot_slopes = None

  def __call__(self, x, nunits):
    if not self._built:
      def _bin_positions(x):
        out_shape = tf.concat((tf.shape(x)[:-1], (nunits, self._nbins)), 0)
        x = tf.reshape(x, out_shape)
        return tf.math.softmax(x, axis=-1) * (
              self._interval_width - self._nbins * self._min_bin_width
              ) + self._min_bin_width

      def _slopes(x):
        out_shape = tf.concat((
          tf.shape(x)[:-1], (nunits, self._nbins - 1)), 0)
        x = tf.reshape(x, out_shape)
        return tf.math.softplus(x) + self._min_slope

      self._bin_widths = tf.keras.layers.Dense(
        nunits * self._nbins, activation=_bin_positions, name='w')
      self._bin_heights = tf.keras.layers.Dense(
        nunits * self._nbins, activation=_bin_positions, name='h')
      self._knot_slopes = tf.keras.layers.Dense(
        nunits * (self._nbins - 1), activation=_slopes, name='s')
      self._built = True

    return tfb.RationalQuadraticSpline(
      bin_widths=self._bin_widths(x),
      bin_heights=self._bin_heights(x),
      knot_slopes=self._knot_slopes(x),
      range_min=self._range_min)

# todo: add nbins and nmber of hidden units and width and range_min as parameters to select with main call
# supports only 2-layers network
# todo: add this
def get_spline_params_network(output_size, width, nbins, net_input, nn_layers):
  def _bin_positions(x):
    out_shape = tf.concat((tf.shape(x)[:-1], (output_size, nbins)), 0)
    x = tf.reshape(x, out_shape)
    return tf.math.softmax(x, axis=-1) * (width - nbins * 1e-3) \
           + 1e-3

  def _slopes(x):
    out_shape = tf.concat((
      tf.shape(x)[:-1], (output_size, nbins - 1)), 0)
    x = tf.reshape(x, out_shape)
    return tf.math.softplus(x) + 1e-3

  #net_input = tfk.Input(shape=(output_size,))

  shared_output = tfk.Sequential([
    tfkl.Dense(nn_layers[0], activation='relu'),
    tfkl.Dense(nn_layers[1], activation='relu'),
    tf.keras.layers.Dense(
      (output_size * nbins) + (output_size * nbins) +
      output_size * (nbins - 1))
  ])(net_input)

  bin_widths, bin_heights, knot_slopes = tfb.Split(
    num_or_size_splits=[output_size * nbins,
                        output_size * nbins,
                        output_size * (nbins - 1)],
    axis=-1
  )(shared_output)

  bin_widths = tfkl.Activation(_bin_positions)(bin_widths)
  bin_heights = tfkl.Activation(_bin_positions)(bin_heights)
  knot_slopes = tfkl.Activation(_slopes)(knot_slopes)

  return [bin_widths, bin_heights, knot_slopes]

def make_splines(input_dim, number_of_bins, nn_layers,
                 b_interval, layers, use_bn=False):

  first_half_dim = input_dim//2
  second_half_dim = input_dim - first_half_dim
  permutation = tf.cast(np.concatenate(
    (np.arange(first_half_dim, input_dim), np.arange(0, first_half_dim))),
                        tf.int32)
  bijector_chain = []
  bijector_chain.append(
    tfb.RealNVP(first_half_dim, bijctor_fn=SplineParams(nbins=number_of_bins,
                                                        interval_width=b_interval,
                                                        range_min=-b_interval / 2))
  )
  if use_bn:
    bijector_chain.append(ActivationNormalization(784))
  for i in range(layers-1):
    if i % 2 == 0:
      dim = second_half_dim
    else:
      dim = first_half_dim
    bijector_chain.append(tfb.Permute(permutation))
    bijector_chain.append(
      tfb.RealNVP(dim, bijctor_fn=SplineParams(nbins=number_of_bins,
                                                          interval_width=b_interval,
                                                          range_min=-b_interval / 2)))
    if use_bn:
      bijector_chain.append(ActivationNormalization(784))
  return bijector_chain