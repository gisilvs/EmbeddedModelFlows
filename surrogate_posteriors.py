import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental

def _mean_field(prior):
  event_shape = prior.event_shape_tensor()
  return tfe.vi.build_affine_surrogate_posterior(event_shape=event_shape, operators='diag')

def _multivariate_normal(prior):
  event_shape = prior.event_shape_tensor()
  return tfe.vi.build_affine_surrogate_posterior(event_shape=event_shape, operators='tril')

def _asvi(prior):
  return tfe.vi.build_asvi_surrogate_posterior(prior)

def _iaf(prior):
  event_shape = prior.event_shape_tensor()
  flat_event_shape = tf.nest.flatten(event_shape)
  flat_event_size = tf.nest.map_structure(tf.reduce_prod, flat_event_shape)
  event_space_bijector = prior.experimental_default_event_space_bijector()

  num_iafs = 2
  base_distribution = tfd.Sample(
    tfd.Normal(0., 1.), sample_shape=[tf.reduce_sum(flat_event_size)])
  iaf_bijectors = [
    tfb.Invert(tfb.MaskedAutoregressiveFlow(
      shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
        params=2, hidden_units=[512, 512], activation='relu'))) #todo: do we need relu here?
    for _ in range(num_iafs)
  ]

  split = tfb.Split(flat_event_size)
  unflatten_bijector = tfb.Restructure(
    tf.nest.pack_sequence_as(
      event_shape, range(len(flat_event_shape))))
  reshape_bijector = tfb.JointMap(
    tf.nest.map_structure(tfb.Reshape, flat_event_shape))

  iaf_surrogate_posterior = tfd.TransformedDistribution(
    base_distribution,
    bijector=tfb.Chain([
                         event_space_bijector,
                         # constrain the surrogate to the support of the prior
                         unflatten_bijector,
                         # pack the reshaped components into the `event_shape` structure of the prior
                         reshape_bijector,
                         # reshape the vector-valued components to match the shapes of the prior components
                         split] +  # Split the samples into components of the same size as the prior components
                       iaf_bijectors
                       # Apply a flow model to the Tensor-valued standard Normal distribution
                       ))

  return iaf_surrogate_posterior


def get_surrogate_posterior(prior, surrogate_posterior_name):

  if surrogate_posterior_name == 'mean_field':
    return _mean_field(prior)

  elif surrogate_posterior_name == 'multivariate_normal':
    return _multivariate_normal(prior)

  elif surrogate_posterior_name == "asvi":
    return _asvi(prior)

  elif surrogate_posterior_name == "iaf":
    return _iaf(prior)

  elif surrogate_posterior_name == "highway_flow":
    return _iaf(prior)