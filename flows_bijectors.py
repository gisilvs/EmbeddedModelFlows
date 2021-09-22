import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import prefer_static as ps

tfb = tfp.bijectors
tfd = tfp.distributions


class SplineParams(tf.Module):

  def __init__(self, nbins=32):
    self._nbins = nbins
    self._built = False
    self._bin_widths = None
    self._bin_heights = None
    self._knot_slopes = None

  def __call__(self, x, nunits):
    if not self._built:
      def _bin_positions(x):
        out_shape = tf.concat((tf.shape(x)[:-1], (nunits, self._nbins)), 0)
        x = tf.reshape(x, out_shape)
        return tf.math.softmax(x, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2

      def _slopes(x):
        out_shape = tf.concat((
          tf.shape(x)[:-1], (nunits, self._nbins - 1)), 0)
        x = tf.reshape(x, out_shape)
        return tf.math.softplus(x) + 1e-2

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
      knot_slopes=self._knot_slopes(x))


def build_rqs(nbins, num_flow_layers):
  splines = [SplineParams(nbins=nbins) for _ in range(num_flow_layers)]
  stack = tfb.Identity()
  for i in range(2):
    stack = tfb.RealNVP(i, bijector_fn=splines[i])(stack)
  return [stack]


def build_iaf_bijector(num_hidden_units,
                       ndims,
                       activation_fn,
                       dtype,
                       num_flow_layers=2, is_iaf=True):
  make_swap = lambda: tfb.Permute(ps.range(ndims - 1, -1, -1))

  def make_maf():
    net = tfb.AutoregressiveNetwork(
      2,
      hidden_units=[num_hidden_units, num_hidden_units],
      activation=activation_fn,
      dtype=dtype)

    maf = tfb.MaskedAutoregressiveFlow(
      bijector_fn=lambda x: tfb.Chain(
        [tfb.Shift(net(x)[Ellipsis, 0]),  # pylint: disable=g-long-lambda
         tfb.Scale(log_scale=net(x)[Ellipsis, 1])]))

    if is_iaf:
      maf = tfb.Invert(maf)
    # To track the variables
    maf._net = net  # pylint: disable=protected-access
    return maf

  iaf_bijector = [make_maf()]
  '''if not is_iaf:
    iaf_bijector.append(tfb.BatchNormalization())'''
  for _ in range(num_flow_layers - 1):
    iaf_bijector.extend([make_swap(), make_maf()])
    '''if not is_iaf:
      iaf_bijector.append(tfb.BatchNormalization())'''

  return iaf_bijector


def build_real_nvp_bijector(num_hidden_units,
                            ndims,
                            num_flow_layers=2):
  def make_rnvp(num_masked):
    rnvp = tfb.RealNVP(
      num_masked,
      shift_and_log_scale_fn=tfb.real_nvp_default_template(
        hidden_layers=[num_hidden_units, num_hidden_units]))

    return rnvp

  d = ndims // 2

  rnvp_bijector = [make_rnvp(d)]
  for i in range(num_flow_layers - 1):
    # rnvp_bijector.append(tfb.Permute(permutation=[1,0]))
    if i % 2 == 0:
      rnvp_bijector.append(make_rnvp(-d))
    else:
      rnvp_bijector.append(make_rnvp(d))

  return rnvp_bijector
