import tensorflow as tf
import tensorflow_probability as tfp

from bijectors.masked_autoregressive import build_iaf_bijector
from bijectors.splines import make_splines
from utils.utils import get_prior_matching_bijectors_and_event_dims

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental
tfp_util = tfp.util


def normalizing_flows(prior, flow_name, flow_params):
  event_shape, flat_event_shape, flat_event_size, ndims, dtype, prior_matching_bijectors = get_prior_matching_bijectors_and_event_dims(
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
  if flow_name == 'splines':
    flow_bijector = make_splines(**flow_params)
  nf = tfd.TransformedDistribution(
    base_distribution,
    bijector=tfb.Chain(prior_matching_bijectors +
                       flow_bijector
                       ))
  return nf
