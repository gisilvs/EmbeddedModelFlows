import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static as ps

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tf.keras
tfkl = tfk.layers


def build_iaf_bijector(num_hidden_units,
                       ndims,
                       activation_fn,
                       dtype,
                       num_flow_layers=2, is_iaf=True, swap=True,
                       use_bn=False):
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
  if use_bn:
    iaf_bijector.append(ActivationNormalization(784))
  for _ in range(num_flow_layers - 1):
    if swap:
      iaf_bijector.extend([make_swap()])
    iaf_bijector.extend([make_maf()])
    if use_bn:
      iaf_bijector.append(ActivationNormalization(784))

  return iaf_bijector
