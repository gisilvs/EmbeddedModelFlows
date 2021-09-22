import tensorflow as tf
import tensorflow_probability as tfp

from flows_bijectors import build_iaf_bijector, build_real_nvp_bijector, build_rqs
from gate_bijector import GateBijector, GateBijectorForNormal
from mixture_of_gaussian_bijector import MixtureOfGaussians, InverseMixtureOfGaussians
from tensorflow_probability.python.internal import prefer_static as ps

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental
tfp_util = tfp.util

# Global dict (DANGEROUS)
residual_fraction_vars = {}


def get_residual_fraction(dist):
  dist_name = dist.parameters['name']
  if dist_name not in residual_fraction_vars:
    # print("CREATING VARIABLE")
    # print(f'{dist_name}')
    bij = tfb.Chain([tfb.Sigmoid(), tfb.Scale(100)])
    residual_fraction_vars[dist_name] = tfp.util.TransformedVariable(0.999,
                                                                     bijector=bij,
                                                                     name='residual_fraction')
  return residual_fraction_vars[dist_name]


# todo: broken with radon, probably need to fix sample and/or independent
stdnormal_bijector_fns = {
  tfd.Gamma: lambda d: tfd.ApproxGammaFromNormal(d.concentration,
                                                 d._rate_parameter()),
  tfd.Normal: lambda d: tfb.Shift(d.loc)(tfb.Scale(d.scale)),
  tfd.HalfNormal: lambda d: tfb.Softplus()(tfb.Scale(d.scale)),
  tfd.MultivariateNormalDiag: lambda d: tfb.Shift(d.loc)(tfb.Scale(d.scale)),
  tfd.MultivariateNormalTriL: lambda d: tfb.Shift(d.loc)(
    tfb.ScaleTriL(d.scale_tril)),
  tfd.TransformedDistribution: lambda d: d.bijector(
    _bijector_from_stdnormal(d.distribution)),
  tfd.Uniform: lambda d: tfb.Shift(d.low)(
    tfb.Scale(d.high - d.low)(tfb.NormalCDF())),
  tfd.Sample: lambda d: _bijector_from_stdnormal_sample(d.distribution),
  tfd.Independent: lambda d: _bijector_from_stdnormal(d.distribution),
  tfd.MixtureSameFamily: lambda d: tfb.Chain([InverseMixtureOfGaussians(d), tfb.NormalCDF()])
}

gated_stdnormal_bijector_fns = {
  tfd.Gamma: lambda d: tfd.ApproxGammaFromNormal(d.concentration,
                                                 d._rate_parameter()),
  # using specific bijector for normal, use next line for generic one
  tfd.Normal: lambda d: GateBijectorForNormal(d.loc, d.scale, get_residual_fraction(d)),
  # tfd.Normal: lambda d: GateBijector(tfb.Shift(d.loc)(tfb.Scale(d.scale)),
                                     # get_residual_fraction(d)),
  tfd.HalfNormal: lambda d: GateBijector(tfb.Softplus()(tfb.Scale(d.scale)),
                                         get_residual_fraction(d)),
  tfd.MultivariateNormalDiag: lambda d: GateBijector(
    tfb.Shift(d.loc)(tfb.Scale(d.scale)), get_residual_fraction(d)),
  tfd.MultivariateNormalTriL: lambda d: GateBijector(tfb.Shift(d.loc)(
    tfb.ScaleTriL(d.scale_tril)), get_residual_fraction(d)),
  tfd.TransformedDistribution: lambda d: d.bijector(
    _gated_bijector_from_stdnormal(d.distribution)),
  tfd.Uniform: lambda d: GateBijector(tfb.Shift(d.low)(
    tfb.Scale(d.high - d.low)(tfb.NormalCDF())), get_residual_fraction(d)),
  tfd.Sample: lambda d: _gated_bijector_from_stdnormal_sample(d.distribution),
  tfd.Independent: lambda d: _gated_bijector_from_stdnormal(d.distribution)
}

stdnormal_bijector_sample_fns = {
  tfd.Normal: lambda d: tfb.Shift(tf.reshape(d.loc, [-1, 1]))(
    tfb.Scale(tf.reshape(d.scale, [-1, 1])))
}

gated_stdnormal_bijector_sample_fns = {
  tfd.Normal: lambda d: GateBijector(tfb.Shift(tf.reshape(d.loc, [-1, 1]))(
    tfb.Scale(tf.reshape(d.scale, [-1, 1]))), get_residual_fraction(d))
}


def _bijector_from_stdnormal_sample(dist):
  fn = stdnormal_bijector_sample_fns[type(dist)]
  return fn(dist)


def _gated_bijector_from_stdnormal_sample(dist):
  fn = gated_stdnormal_bijector_sample_fns[type(dist)]
  return fn(dist)


def _bijector_from_stdnormal(dist):
  fn = stdnormal_bijector_fns[type(dist)]
  return fn(dist)


def _gated_bijector_from_stdnormal(dist):
  fn = gated_stdnormal_bijector_fns[type(dist)]
  return fn(dist)


class AutoFromNormal(tfd.joint_distribution._DefaultJointBijector):

  def __init__(self, dist):
    return super().__init__(dist, bijector_fn=_bijector_from_stdnormal)


class GatedAutoFromNormal(tfd.joint_distribution._DefaultJointBijector):

  def __init__(self, dist):
    return super().__init__(dist, bijector_fn=_gated_bijector_from_stdnormal)


def _get_prior_matching_bijectors_and_event_dims(prior):
  event_shape = prior.event_shape_tensor()
  flat_event_shape = tf.nest.flatten(event_shape)
  flat_event_size = tf.nest.map_structure(tf.reduce_prod, flat_event_shape)
  try:
    event_space_bijector = prior.experimental_default_event_space_bijector()
  except:
    event_space_bijector = None

  split_bijector = tfb.Split(flat_event_size)
  unflatten_bijector = tfb.Restructure(
    tf.nest.pack_sequence_as(
      event_shape, range(len(flat_event_shape))))
  reshape_bijector = tfb.JointMap(
    tf.nest.map_structure(tfb.Reshape, flat_event_shape,
                          [x[tf.newaxis] for x in flat_event_size]))

  if event_space_bijector:

    prior_matching_bijectors = [event_space_bijector, unflatten_bijector,
                                reshape_bijector, split_bijector]

  else:
    prior_matching_bijectors = [unflatten_bijector,
                                reshape_bijector, split_bijector]

  dtype = tf.nest.flatten(prior.dtype)[0]

  return event_shape, flat_event_shape, flat_event_size, int(
    tf.reduce_sum(flat_event_size)), dtype, prior_matching_bijectors


def _mean_field(prior):
  return tfe.vi.build_asvi_surrogate_posterior(prior, mean_field=True)


def _multivariate_normal(prior):
  def make_trainable_linear_operator_tril(
      dim,
      scale_initializer=1e-1,
      diag_bijector=None,
      diag_shift=1e-5,
      dtype=tf.float32):
    """Build a trainable lower triangular linop."""
    scale_tril_bijector = tfb.FillScaleTriL(
      diag_bijector, diag_shift=diag_shift)
    flat_initial_scale = tf.zeros((dim * (dim + 1) // 2,), dtype=dtype)
    initial_scale_tril = tfb.FillScaleTriL(
      diag_bijector=tfb.Identity(), diag_shift=scale_initializer)(
      flat_initial_scale)
    return tf.linalg.LinearOperatorLowerTriangular(
      tril=tfp_util.TransformedVariable(
        initial_scale_tril, bijector=scale_tril_bijector))

  event_shape, flat_event_shape, flat_event_size, ndims, dtype, prior_matching_bijectors = _get_prior_matching_bijectors_and_event_dims(
    prior)
  base_dist = tfd.Sample(
    tfd.Normal(tf.zeros([], dtype), 1.), sample_shape=[ndims])
  op = make_trainable_linear_operator_tril(ndims)

  prior_matching_bijectors.extend(
    [tfb.Shift(tf.Variable(tf.zeros([ndims], dtype=dtype))),
     tfb.ScaleMatvecLinearOperator(op)])

  return tfd.TransformedDistribution(base_dist,
                                     tfb.Chain(prior_matching_bijectors))


def _asvi(prior):
  return tfe.vi.build_asvi_surrogate_posterior(prior)


def _normalizing_flows(prior, flow_name, flow_params):
  event_shape, flat_event_shape, flat_event_size, ndims, dtype, prior_matching_bijectors = _get_prior_matching_bijectors_and_event_dims(
    prior)

  base_distribution = tfd.Sample(
    tfd.Normal(tf.zeros([], dtype=dtype), 1.), sample_shape=[ndims])

  if flow_name == 'iaf':
    flow_params['dtype'] = dtype
    flow_params['ndims'] = ndims
    flow_bijector = build_iaf_bijector(**flow_params)
  if flow_name == 'maf':
    flow_params['dtype'] = dtype
    flow_params['ndims'] = ndims
    flow_bijector = build_iaf_bijector(**flow_params)
  if flow_name == 'real_nvp':
    # flow_params['dtype'] = dtype
    flow_params['ndims'] = ndims
    flow_bijector = build_real_nvp_bijector(**flow_params)
  if flow_name == 'rqs':
    flow_bijector = build_rqs(**flow_params)

  nf_surrogate_posterior = tfd.TransformedDistribution(
    base_distribution,
    bijector=tfb.Chain(prior_matching_bijectors +
                       flow_bijector
                       ))

  return nf_surrogate_posterior

def _normalizing_program(prior, backbone_name, flow_params):
  bijector = AutoFromNormal(prior)
  backbone_surrogate_posterior = get_surrogate_posterior(prior,
                                                         surrogate_posterior_name=backbone_name,
                                                         flow_params=flow_params)
  return tfd.TransformedDistribution(
    distribution=backbone_surrogate_posterior,
    bijector=bijector
  )

def _sandwich_maf_normalizing_program(prior, num_layers_per_flow=1, is_gated=False):
  event_shape, flat_event_shape, flat_event_size, ndims, dtype, prior_matching_bijectors = _get_prior_matching_bijectors_and_event_dims(
    prior)

  base_distribution = tfd.Sample(
    tfd.Normal(tf.zeros([], dtype=dtype), 1.), sample_shape=[ndims])

  flow_params = {'activation_fn': tf.nn.relu}
  flow_params['dtype'] = dtype
  flow_params['ndims'] = ndims
  flow_params['num_flow_layers'] = num_layers_per_flow
  flow_params['num_hidden_units'] = 512
  flow_params['is_iaf'] = False
  flow_bijector_pre = build_iaf_bijector(**flow_params)
  flow_bijector_post = build_iaf_bijector(**flow_params)
  make_swap = lambda: tfb.Permute(ps.range(ndims - 1, -1, -1))
  if is_gated:
    normalizing_program = GatedAutoFromNormal(prior)
  else:
    normalizing_program = AutoFromNormal(prior)
  prior_matching_bijectors = tfb.Chain(prior_matching_bijectors)

  bijector = tfb.Chain([prior_matching_bijectors,
                        flow_bijector_post[0],
                        tfb.Chain([tfb.Invert(prior_matching_bijectors),
                        normalizing_program,
                        prior_matching_bijectors]),
                        make_swap(),
                        flow_bijector_pre[0]])

  backbone_surrogate_posterior = tfd.TransformedDistribution(
    distribution=base_distribution,
    bijector=bijector
  )

  return backbone_surrogate_posterior

def bottom_np_maf(prior):
  event_shape, flat_event_shape, flat_event_size, ndims, dtype, prior_matching_bijectors = _get_prior_matching_bijectors_and_event_dims(
    prior)

  base_distribution = tfd.Sample(
    tfd.Normal(tf.zeros([], dtype=dtype), 1.), sample_shape=[ndims])

  flow_params = {'activation_fn': tf.nn.relu}
  flow_params['dtype'] = dtype
  flow_params['ndims'] = ndims
  flow_params['num_flow_layers'] = 1
  flow_params['num_hidden_units'] = 512
  flow_params['is_iaf'] = False
  flow_bijector_pre = build_iaf_bijector(**flow_params)
  flow_bijector_post = build_iaf_bijector(**flow_params)
  normalizing_program = AutoFromNormal(prior)
  prior_matching_bijectors = tfb.Chain(prior_matching_bijectors)

  bijector = tfb.Chain([
    prior_matching_bijectors,
    flow_bijector_post[0],
    flow_bijector_pre[0],
    tfb.Chain([tfb.Invert(prior_matching_bijectors),
               normalizing_program,
               prior_matching_bijectors])
  ])

  backbone_surrogate_posterior = tfd.TransformedDistribution(
    distribution=base_distribution,
    bijector=bijector
  )

  return backbone_surrogate_posterior

def _gated_normalizing_program(prior, backbone_name, flow_params):
  '''for d in prior._get_single_sample_distributions():
    if type(d) == tfd.Independent or type(d) == tfd.Sample:
      d.distribution._residual_fraction = tfp.util.TransformedVariable(0.98, bijector=tfb.Sigmoid())
    else:
      d._residual_fraction = tfp.util.TransformedVariable(0.98, bijector=tfb.Sigmoid())'''

  backbone_surrogate_posterior = get_surrogate_posterior(prior,
                                                         surrogate_posterior_name=backbone_name,
                                                         flow_params=flow_params)

  '''for d in backbone_surrogate_posterior._get_single_sample_distributions():
    if type(d) == tfd.Independent or type(d) == tfd.Sample:
      d.distribution._residual_fraction = tfp.util.TransformedVariable(0.98, bijector=tfb.Sigmoid())
    else:
      d._residual_fraction = tfp.util.TransformedVariable(0.98, bijector=tfb.Sigmoid())'''

  bijector = GatedAutoFromNormal(prior)
  return tfd.TransformedDistribution(
    distribution=backbone_surrogate_posterior,
    bijector=bijector
  )


def get_surrogate_posterior(prior, surrogate_posterior_name,
                            backnone_name=None, flow_params={}):
  # Needed to reset the gates if running several experiments sequentially
  global residual_fraction_vars
  residual_fraction_vars = {}

  if surrogate_posterior_name == 'mean_field':
    return _mean_field(prior)

  elif surrogate_posterior_name == 'multivariate_normal':
    return _multivariate_normal(prior)

  elif surrogate_posterior_name == "asvi":
    return _asvi(prior)

  elif surrogate_posterior_name == "iaf":
    flow_params['num_flow_layers'] = 2
    flow_params['num_hidden_units'] = 512
    if 'activation_fn' not in flow_params:
      flow_params['activation_fn'] = tf.math.tanh
    return _normalizing_flows(prior, flow_name='iaf', flow_params=flow_params)

  elif surrogate_posterior_name == "maf":
    flow_params['num_flow_layers'] = 2
    flow_params['num_hidden_units'] = 512
    flow_params['is_iaf'] = False
    if 'activation_fn' not in flow_params:
      flow_params['activation_fn'] = tf.math.tanh
    return _normalizing_flows(prior, flow_name='maf', flow_params=flow_params)

  elif surrogate_posterior_name == "real_nvp":
    flow_params = {
      'num_flow_layers': 2,
      'num_hidden_units': 512
    }
    return _normalizing_flows(prior, flow_name='real_nvp',
                              flow_params=flow_params)

  elif surrogate_posterior_name == "rqs":
    return _normalizing_flows(prior, flow_name='rqs',
                              flow_params=flow_params)

  elif surrogate_posterior_name == "normalizing_program":
    if backnone_name == 'iaf':
      flow_params = {'activation_fn': tf.nn.relu}
    elif backnone_name == 'maf':
      flow_params = {'activation_fn': tf.nn.relu}
    else:
      flow_params = {}
    return _normalizing_program(prior, backbone_name=backnone_name,
                                flow_params=flow_params)

  elif surrogate_posterior_name == "gated_normalizing_program":
    if backnone_name == 'iaf':
      flow_params = {'activation_fn': tf.nn.relu}
    elif backnone_name == 'maf':
      flow_params = {'activation_fn': tf.nn.relu}
    else:
      flow_params = {}
    return _gated_normalizing_program(prior, backbone_name=backnone_name,
                                      flow_params=flow_params)
