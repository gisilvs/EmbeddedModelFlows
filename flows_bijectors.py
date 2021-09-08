import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import prefer_static as ps

tfb = tfp.bijectors
tfd = tfp.distributions


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
  if not is_iaf:
    iaf_bijector.append(tfb.Invert(tfb.BatchNormalization()))
  for _ in range(num_flow_layers - 1):
    iaf_bijector.extend([make_swap(), make_maf()])
    if not is_iaf:
      iaf_bijector.append(tfb.Invert(tfb.BatchNormalization()))

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

  d = ndims//2

  rnvp_bijector = [make_rnvp(d)]
  for i in range(num_flow_layers - 1):
    #rnvp_bijector.append(tfb.Permute(permutation=[1,0]))
    if i % 2 == 0:
      rnvp_bijector.append(make_rnvp(-d))
    else:
      rnvp_bijector.append(make_rnvp(d))

  return rnvp_bijector
