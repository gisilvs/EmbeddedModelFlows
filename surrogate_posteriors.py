import tensorflow as tf
import tensorflow_probability as tfp
from flows_bijectors import build_highway_flow_bijector, build_iaf_biector

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental

stdnormal_bijector_fns = {
    tfd.Gamma: lambda d: tfd.ApproxGammaFromNormal(d.concentration, d._rate_parameter()),
    tfd.Normal: lambda d: tfb.Shift(d.loc)(tfb.Scale(d.scale)),
    tfd.MultivariateNormalDiag: lambda d: tfb.Shift(d.loc)(tfb.Scale(d.scale)),
    tfd.MultivariateNormalTriL: lambda d: tfb.Shift(d.loc)(tfb.ScaleTriL(d.scale_tril)),
    tfd.TransformedDistribution: lambda d: d.bijector(_bijector_from_stdnormal(d.distribution)),
    tfd.Uniform: lambda d: tfb.Shift(d.low)(tfb.Scale(d.high - d.low)(tfb.NormalCDF())),
    tfd.Sample: lambda d: _bijector_from_stdnormal(d.distribution),
    tfd.Independent: lambda d: _bijector_from_stdnormal(d.distribution)
}

def _bijector_from_stdnormal(dist):
  fn = stdnormal_bijector_fns[type(dist)]
  return fn(dist)


class AutoFromNormal(tfd.joint_distribution._DefaultJointBijector):

  def __init__(self, dist):
    return super().__init__(dist, bijector_fn=_bijector_from_stdnormal)

def _mean_field(prior):
  event_shape = prior.event_shape_tensor()
  return tfe.vi.build_affine_surrogate_posterior(event_shape=event_shape, operators='diag')

def _multivariate_normal(prior):
  event_shape = prior.event_shape_tensor()
  return tfe.vi.build_affine_surrogate_posterior(event_shape=event_shape, operators='tril')

def _asvi(prior):
  return tfe.vi.build_asvi_surrogate_posterior(prior)

def _normalizing_flows(prior, flow_name, flow_params):
  event_shape = prior.event_shape_tensor()
  flat_event_shape = tf.nest.flatten(event_shape)
  flat_event_size = tf.nest.map_structure(tf.reduce_prod, flat_event_shape)
  event_space_bijector = prior.experimental_default_event_space_bijector()

  base_distribution = tfd.Sample(
    tfd.Normal(0., 1.), sample_shape=[tf.reduce_sum(flat_event_size)])

  split = tfb.Split(flat_event_size)
  unflatten_bijector = tfb.Restructure(
    tf.nest.pack_sequence_as(
      event_shape, range(len(flat_event_shape))))
  reshape_bijector = tfb.JointMap(
    tf.nest.map_structure(tfb.Reshape, flat_event_shape))

  if flow_name=='iaf':
    flow_bijector = build_iaf_biector(**flow_params)
  elif flow_name=='highway_flow':
    flow_params['width'] = int(tf.reduce_sum(flat_event_size))
    flow_params['gate_first_n'] = flow_params['width']
    flow_bijector = build_highway_flow_bijector(**flow_params)

  nf_surrogate_posterior = tfd.TransformedDistribution(
    base_distribution,
    bijector=tfb.Chain([
                         event_space_bijector,
                         # constrain the surrogate to the support of the prior
                         unflatten_bijector,
                         # pack the reshaped components into the `event_shape` structure of the prior
                         reshape_bijector,
                         # reshape the vector-valued components to match the shapes of the prior components
                         split] +  # Split the samples into components of the same size as the prior components
                       flow_bijector
                       # Apply a flow model to the Tensor-valued standard Normal distribution
                       ))

  return nf_surrogate_posterior


def _normalizing_program(prior, backbone_name):
  backbone_surrogate_posterior = get_surrogate_posterior(prior, surrogate_posterior_name=backbone_name)
  bijector = AutoFromNormal(prior)
  return tfd.TransformedDistribution(
    distribution=backbone_surrogate_posterior,
    bijector=bijector
  )




def get_surrogate_posterior(prior, surrogate_posterior_name, backnone_name=None):

  if surrogate_posterior_name == 'mean_field':
    return _mean_field(prior)

  elif surrogate_posterior_name == 'multivariate_normal':
    return _multivariate_normal(prior)

  elif surrogate_posterior_name == "asvi":
    return _asvi(prior)

  elif surrogate_posterior_name == "small_iaf":
    flow_params = {
      'num_iaf':2,
      'hidden_units':8
    }
    return _normalizing_flows(prior, flow_name='iaf', flow_params=flow_params)

  elif surrogate_posterior_name == "large_iaf":
    flow_params = {
      'num_iaf': 2,
      'hidden_units': 512
    }
    return _normalizing_flows(prior, flow_name='iaf', flow_params=flow_params)

  elif surrogate_posterior_name == "highway_flow":
    flow_params = {
      'num_layers': 3,
      'residual_fraction_initial_value':0.98
    }
    return _normalizing_flows(prior, flow_name='highway_flow', flow_params=flow_params)

  elif surrogate_posterior_name == "normalizing_program":
    return _normalizing_program(prior, backbone_name=backnone_name)


