import numpy as np
import tensorflow
import tensorflow as tf
import tensorflow_probability as tfp

from .actnorm import ActivationNormalization

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental
tfp_util = tfp.util
tfkl = tf.keras.layers
tfk = tensorflow.keras


# TODO: refactor to tfp splines

class NN_Spline(tfkl.Layer):
  def __init__(self, layers, k_dim, remaining_dims, first_d_dims,
               activation="relu"):
    super(NN_Spline, self).__init__(name="nn")
    self.k_dim = k_dim
    layer_list = []
    layer_list.append(
      tfkl.Dense(layers[0], activation=activation, input_dim=first_d_dims,
                 dtype=tf.float32, name=f'0_layer'))
    for i, hidden in enumerate(layers[1:]):
      layer_list.append(
        tfkl.Dense(hidden, activation=activation, dtype=tf.float32,
                   name=f'{i + 1}_layer'))
    layer_list.append(
      tfkl.Dense(remaining_dims * (3 * k_dim - 1), dtype=tf.float32,
                 name='last_layer'))
    self.layer_list = layer_list

  def call(self, x):
    y = x
    for layer in self.layer_list:
      y = layer(y)
    return y


class NeuralSplineFlow(tfb.Bijector):
  """
  Implementation of a Neural Spline Flows by Durkan et al. [1].
  :param n_dims: The dimension of the vector-sized input. Each individual input should be a vector with d_dim dimensions.
  :param number_of_bins: Number of bins to create the spline
  :param nn_layers: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
  :param b_interval: Interval to define the spline function. Spline function behaves as identity outside of the interval
  :param d_dim: The number of dimensions to create the parameters of the spline. (d_dim-1) dims are used to create the parameters as in paper.
  :param simetric_interval: If this is true we have a interval of [-b_interval, b_interval]. [0, 2*b_interval] if false.
  """

  def __init__(self, input_dim, d_dim, b_interval, number_of_bins=5,
               nn_layers=[16, 16], simetric_interval: bool = True,
               validate_args: bool = False, name="neural_spline_flow"):
    super(NeuralSplineFlow, self).__init__(
      validate_args=validate_args, forward_min_event_ndims=1, name=name
    )

    self.event_ndims = 1
    self.total_input_dim = input_dim
    self.first_d_dims = d_dim - 1
    self.remaining_dims = input_dim - self.first_d_dims
    self.number_of_bins = number_of_bins
    self.number_of_knots = number_of_bins + 1
    self.b_interval = tf.constant(b_interval, dtype=tf.float32)
    self.nn = NN_Spline(layers=nn_layers, k_dim=self.number_of_bins,
                        first_d_dims=self.first_d_dims,
                        remaining_dims=self.remaining_dims)
    x = tf.keras.Input(self.first_d_dims, dtype=tf.float32)
    output = self.nn(x)
    self.min_bin_width = 1e-3  # maximum number of bins 1/1e-3 then...
    self.nn_model = tfk.Model(x, output, name="nn")
    self.simetric_interval = simetric_interval

  # some calculation could be done in one-line of code but it was preferred to explicitly write them
  # for easy debugging purposes during the development and also to give an understanding of the implementations of the terms in the paper
  # to the reader

  def return_identity(self, x):
    return x

  def return_forward_result(self, x_d_to_D, input_mask, x_1_to_d,
                            intervals_for_func):
    output = tf.zeros(tf.shape(x_d_to_D))
    input_mask_indexes = tf.where(input_mask)
    neg_input_mask_indexes = tf.where(~input_mask)
    thetas = self._produce_thetas(x_1_to_d)
    thetas_1, thetas_2, thetas_3 = self._get_thetas(thetas,
                                                    input_mask_indexes)
    interval_indices = input_mask_indexes[:, 1]

    input_for_spline = x_d_to_D[input_mask]
    intervals_for_input = tf.gather(intervals_for_func, interval_indices)
    x_bin_sizes = self._bins(thetas_1, intervals_for_input)
    knot_xs = self._knots(x_bin_sizes, intervals_for_input)
    y_bin_sizes = self._bins(thetas_2, intervals_for_input)
    knot_ys = self._knots(y_bin_sizes, intervals_for_input)
    derivatives = self._derivatives(thetas_3)
    locs = self._knots_locations(input_for_spline, knot_xs)
    floor_indices = self._indices(locs - 1)
    ceil_indices = self._indices(locs)
    xi_values = self._xi_values(input_for_spline, knot_xs, x_bin_sizes,
                                floor_indices)
    s_values = self._s_values(y_bin_sizes, x_bin_sizes)
    forward_val = self._g_function(input_for_spline, floor_indices,
                                   ceil_indices, xi_values, s_values,
                                   y_bin_sizes, derivatives, knot_ys)
    output = tf.tensor_scatter_nd_update(
      tf.dtypes.cast(tf.expand_dims(output, 2), dtype=tf.float32),
      input_mask_indexes, tf.expand_dims(
        tf.dtypes.cast(tf.transpose(forward_val), dtype=tf.float32), 1))
    output = tf.tensor_scatter_nd_update(output, neg_input_mask_indexes,
                                         tf.expand_dims(x_d_to_D[~input_mask],
                                                        1))
    return output

  def return_inverse_result(self, y_d_to_D, input_mask, y_1_to_d,
                            intervals_for_func):
    output = tf.zeros(tf.shape(y_d_to_D), dtype=tf.float32)
    input_mask_indexes = tf.where(input_mask)
    neg_input_mask_indexes = tf.where(~input_mask)
    thetas = self._produce_thetas(y_1_to_d)
    thetas_1, thetas_2, thetas_3 = self._get_thetas(thetas,
                                                    input_mask_indexes)
    input_for_inverse = y_d_to_D[input_mask]
    interval_indices = input_mask_indexes[:, 1]

    intervals_for_input = tf.gather(intervals_for_func, interval_indices)
    x_bin_sizes = self._bins(thetas_1, intervals_for_input)
    knot_xs = self._knots(x_bin_sizes, intervals_for_input)
    y_bin_sizes = self._bins(thetas_2, intervals_for_input)
    knot_ys = self._knots(y_bin_sizes, intervals_for_input)
    derivatives = self._derivatives(thetas_3)
    locs = self._knots_locations(input_for_inverse, knot_ys)
    floor_indices = self._indices(locs - 1)
    ceil_indices = self._indices(locs)
    s_values = self._s_values(y_bin_sizes, x_bin_sizes)

    inverse_val = self._inverse_g_function(input_for_inverse, floor_indices,
                                           ceil_indices, s_values,
                                           y_bin_sizes, derivatives, knot_ys,
                                           knot_xs, x_bin_sizes)
    output = tf.tensor_scatter_nd_update(
      tf.dtypes.cast(tf.expand_dims(output, 2), dtype=tf.float32),
      input_mask_indexes, tf.expand_dims(
        tf.dtypes.cast(tf.transpose(inverse_val), dtype=tf.float32), 1))
    output = tf.tensor_scatter_nd_update(
      tf.dtypes.cast(output, dtype=tf.float32), neg_input_mask_indexes,
      tf.dtypes.cast(tf.expand_dims(y_d_to_D[~input_mask], 1), tf.float32))
    return tf.concat([tf.dtypes.cast(y_1_to_d, tf.float32),
                      tf.dtypes.cast(tf.squeeze(output, -1), tf.float32)],
                     axis=-1)

  def return_identity_log_det(self):
    return tf.constant(0.0, dtype=tf.float32)

  def return_result_log_det(self, x, input_mask, x_1_to_d, intervals_for_func, \
                            x_d_to_D):
    input_mask_indexes = tf.where(input_mask)
    neg_input_mask_indexes = tf.where(~input_mask)
    thetas = self._produce_thetas(x_1_to_d)
    thetas_1, thetas_2, thetas_3 = self._get_thetas(thetas,
                                                    input_mask_indexes)
    interval_indices = input_mask_indexes[:, 1]
    intervals_for_input = tf.gather(intervals_for_func, interval_indices)
    input_for_derivative = x_d_to_D[input_mask]
    x_bin_sizes = self._bins(thetas_1, intervals_for_input)
    knot_xs = self._knots(x_bin_sizes, intervals_for_input)
    y_bin_sizes = self._bins(thetas_2, intervals_for_input)
    knot_ys = self._knots(y_bin_sizes, intervals_for_input)
    derivatives = self._derivatives(thetas_3)
    locs = self._knots_locations(input_for_derivative, knot_xs)
    floor_indices = self._indices(locs - 1)
    ceil_indices = self._indices(locs)
    s_values = self._s_values(y_bin_sizes, x_bin_sizes)
    xi_values = self._xi_values(input_for_derivative, knot_xs, x_bin_sizes,
                                floor_indices)
    dervs = self._derivative_of_g_func(input_for_derivative, floor_indices,
                                       ceil_indices, xi_values, s_values,
                                       derivatives)
    output = tf.ones(tf.shape(x), dtype=tf.float32)
    squeezed = tf.tensor_scatter_nd_update(
      tf.dtypes.cast(tf.expand_dims(output, 2), dtype=tf.float32),
      input_mask_indexes,
      tf.expand_dims(tf.transpose(tf.dtypes.cast(dervs, dtype=tf.float32)),
                     1))
    output = tf.squeeze(squeezed)
    log_dervs = tf.math.log(output)
    log_det_sum = tf.reduce_sum(log_dervs, axis=1)
    return log_det_sum

  def _produce_thetas(self, x):
    thetas = self.nn_model(x)
    thetas = tf.reshape(thetas, [tf.shape(x)[0], self.remaining_dims,
                                 3 * self.number_of_bins - 1])
    return thetas

  def _get_thetas(self, thetas, input_mask_indexes):
    thetas_for_input = tf.gather_nd(thetas, input_mask_indexes)
    thetas_1 = thetas_for_input[:, :self.number_of_bins]
    thetas_2 = thetas_for_input[:, self.number_of_bins:2 * self.number_of_bins]
    thetas_3 = thetas_for_input[:, 2 * self.number_of_bins:]
    return thetas_1, thetas_2, thetas_3

  def _bins(self, thetas, intervals):
    normalized_widths = tf.math.softmax(thetas)
    normalized_widths_filled = self.min_bin_width + (
        1 - self.min_bin_width * self.number_of_bins) * normalized_widths
    expanded_widths = normalized_widths_filled * 2 * tf.expand_dims(intervals,
                                                                    1)
    return expanded_widths

  def _knots(self, bins, intervals):
    interval = -1 * tf.expand_dims(intervals, 1)
    b = tf.concat([tf.zeros((tf.shape(bins)[0], 1), dtype=tf.float32),
                   tf.dtypes.cast((tf.math.cumsum(bins, axis=1)), tf.float32)],
                  1) + tf.dtypes.cast(interval,
                                      tf.float32) if self.simetric_interval else tf.concat(
      [tf.zeros((tf.shape(bins)[0], 1), dtype=tf.float32),
       tf.dtypes.cast((tf.math.cumsum(bins, axis=1)), tf.float32)], 1)
    return b

  def _derivatives(self, thetas):
    inner_derivatives = tf.math.softplus(thetas)
    c = tf.concat(
      [tf.ones((tf.shape(inner_derivatives)[0], 1), dtype=tf.float32),
       inner_derivatives,
       tf.ones((tf.shape(inner_derivatives)[0], 1), dtype=tf.float32)], 1)
    return c + self.min_bin_width

  def _s_values(self, y_bins, x_bins):
    y = y_bins / x_bins
    return y

  def _knots_locations(self, x, knot_xs):
    x_binary_mask = tf.cast((tf.expand_dims(x, 1) > knot_xs), tf.int32)
    knot_xs = tf.reduce_sum(x_binary_mask, axis=1)
    return knot_xs

  def _indices(self, locations):
    row_indices = tf.range(tf.shape(locations)[0], dtype=tf.int32)
    z = tf.transpose(tf.stack([row_indices, locations]))
    return z

  def _xi_values(self, x, knot_xs, x_bin_sizes, ind):
    f = (tf.transpose(x) - tf.gather_nd(knot_xs, ind)) / tf.gather_nd(
      x_bin_sizes, ind)
    return f

  def _g_function(self, x, bin_ind, knot_ind, xi_values, s_values, y_bin_sizes,
                  derivatives, knot_ys):
    xi_times_1_minus_xi = xi_values * (1 - xi_values)
    s_k = tf.gather_nd(s_values, bin_ind)
    y_kplus1_minus_y_k = tf.gather_nd(y_bin_sizes, bin_ind)
    xi_square = xi_values ** 2
    d_k = tf.gather_nd(derivatives, bin_ind)
    d_kplus1 = tf.gather_nd(derivatives, knot_ind)
    y_k = tf.gather_nd(knot_ys, bin_ind)
    second_term_nominator = y_kplus1_minus_y_k * (
        s_k * xi_square + d_k * xi_times_1_minus_xi)
    second_term_denominator = s_k + (
        d_kplus1 + d_k - 2 * s_k) * xi_times_1_minus_xi
    forward_val = y_k + second_term_nominator / second_term_denominator
    return forward_val

  def _inverse_g_function(self, input_for_inverse, floor_indices, ceil_indices,
                          s_values, y_bin_sizes, derivatives, knot_ys, knot_xs,
                          x_bin_sizes):
    y_minus_y_k = tf.dtypes.cast(tf.transpose(input_for_inverse),
                                 tf.float32) - tf.dtypes.cast(
      tf.gather_nd(knot_ys, floor_indices), tf.float32)
    s_k = tf.gather_nd(s_values, floor_indices)
    y_kplus1_minus_y_k = tf.gather_nd(y_bin_sizes, floor_indices)
    d_k = tf.gather_nd(derivatives, floor_indices)
    d_kplus1 = tf.gather_nd(derivatives, ceil_indices)
    common_term = y_minus_y_k * (d_kplus1 + d_k - 2 * s_k)
    a = y_kplus1_minus_y_k * (s_k - d_k) + common_term
    b = y_kplus1_minus_y_k * d_k - common_term
    c = -1 * s_k * y_minus_y_k
    b_squared_minus_4ac = b ** 2 - 4 * a * c
    sqrt_b_squared_minus_4ac = tf.math.sqrt(b_squared_minus_4ac)
    denominator = (-1 * b - sqrt_b_squared_minus_4ac)
    xi_x_d_to_D = 2 * c / denominator
    x_d_to_D = xi_x_d_to_D * tf.gather_nd(x_bin_sizes,
                                          floor_indices) + tf.gather_nd(knot_xs,
                                                                        floor_indices)
    return x_d_to_D

  def _derivative_of_g_func(self, x, floor_indices, ceil_indices, xi_values,
                            s_values, derivatives):
    one_minus_xi = (1 - xi_values)
    xi_times_1_minus_xi = xi_values * one_minus_xi
    s_k = tf.gather_nd(s_values, floor_indices)
    one_minus_xi_square = one_minus_xi ** 2
    d_k = tf.gather_nd(derivatives, floor_indices)
    d_kplus1 = tf.gather_nd(derivatives, ceil_indices)
    nominator = s_k ** 2 * (d_kplus1 * (
        xi_values ** 2) + 2 * s_k * xi_times_1_minus_xi + d_k * one_minus_xi_square)
    denominator = (s_k + (d_kplus1 + d_k - 2 * s_k) * xi_times_1_minus_xi) ** 2
    derivative_result = nominator / denominator
    return derivative_result

  def _data_mask(self, x_d_to_D, interval):
    less_than_right_limit_mask = x_d_to_D < interval
    bigger_than_left_limit_mask = x_d_to_D > -1.0 * interval
    input_mask = less_than_right_limit_mask & bigger_than_left_limit_mask
    return input_mask

  def _forward(self, x):
    x_1_to_d, x_d_to_D = x[:, :self.first_d_dims], x[:, self.first_d_dims:]
    # x_d_to_D = tf.constant(x_d_to_D, dtype=tf.float32)
    # x_1_to_d = tf.constant(x_1_to_d, dtype=tf.float32)
    _, intervals_for_func = self.b_interval[
                            :self.first_d_dims], self.b_interval[
                                                 self.first_d_dims:]
    y_1_to_d = x_1_to_d
    input_mask = self._data_mask(x_d_to_D, intervals_for_func)

    r = tf.cond(tf.equal(tf.reduce_any(input_mask), tf.constant(False)),
                lambda: self.return_identity(x), lambda:
                self.return_forward_result(
                  x_d_to_D,
                  input_mask, x_1_to_d,
                  intervals_for_func))
    y = tf.concat([y_1_to_d, tf.squeeze(r, -1)], axis=-1)
    return y

  def _inverse(self, y):
    y_1_to_d, y_d_to_D = y[:, :self.first_d_dims], y[:, self.first_d_dims:]
    _, intervals_for_func = self.b_interval[
                            :self.first_d_dims], self.b_interval[
                                                 self.first_d_dims:]
    input_mask = self._data_mask(y_d_to_D, intervals_for_func)
    return tf.cond(tf.equal(tf.reduce_any(input_mask), tf.constant(False)),
                   lambda: self.return_identity(y),
                   lambda: self.return_inverse_result(y_d_to_D, input_mask,
                                                      y_1_to_d,
                                                      intervals_for_func))

  def _forward_log_det_jacobian(self, x, thetas=None):
    x_1_to_d, x_d_to_D = x[:, :self.first_d_dims], x[:, self.first_d_dims:]
    _, intervals_for_func = self.b_interval[
                            :self.first_d_dims], self.b_interval[
                                                 self.first_d_dims:]
    input_mask = self._data_mask(x_d_to_D, intervals_for_func)

    return tf.cond(tf.equal(tf.reduce_any(input_mask), tf.constant(False)),
                   lambda: self.return_identity_log_det(), lambda:
                   self.return_result_log_det(x, input_mask, x_1_to_d,
                                              intervals_for_func, x_d_to_D))


def make_splines(input_dim, number_of_bins, nn_layers,
                 b_interval, layers, use_bn=False):
  permutation = tf.cast(np.concatenate(
    (np.arange(input_dim / 2, input_dim), np.arange(0, input_dim / 2))),
    tf.int32)
  bijector_chain = []
  bijector_chain.append(
    NeuralSplineFlow(input_dim=input_dim, d_dim=int(input_dim / 2) + 1,
                     number_of_bins=number_of_bins, nn_layers=nn_layers,
                     b_interval=[b_interval for _ in range(input_dim)]))
  if use_bn:
    bijector_chain.append(ActivationNormalization(784))
  for i in range(layers - 1):
    bijector_chain.append(tfb.Permute(permutation))
    bijector_chain.append(
      NeuralSplineFlow(input_dim=input_dim, d_dim=int(input_dim / 2) + 1,
                       number_of_bins=number_of_bins, nn_layers=nn_layers,
                       b_interval=[b_interval for _ in range(input_dim)]))
    if use_bn:
      bijector_chain.append(ActivationNormalization(784))
  return bijector_chain
