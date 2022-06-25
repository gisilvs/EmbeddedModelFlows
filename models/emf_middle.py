import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static as ps

from bijectors.actnorm import ActivationNormalization
from bijectors.gated_structured_layer import GatedStructuredLayer
from bijectors.masked_autoregressive import build_iaf_bijector
from bijectors.splines import make_splines
from bijectors.structured_layer import StructuredLayer
from utils.utils import get_prior_matching_bijectors_and_event_dims

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental
tfp_util = tfp.util


# todo: add flag for gated
def emf_middle(prior, num_layers_per_flow=1,
               use_bn=False, use_gates=False):
  event_shape, flat_event_shape, flat_event_size, ndims, dtype, prior_matching_bijectors = get_prior_matching_bijectors_and_event_dims(
    prior)

  base_distribution = tfd.Sample(
    tfd.Normal(tf.zeros([], dtype=dtype), 1.), sample_shape=[ndims])

  flow_params = {'activation_fn': tf.nn.relu}
  flow_params['dtype'] = tf.float32
  flow_params['ndims'] = ndims
  flow_params['num_flow_layers'] = num_layers_per_flow
  flow_params['num_hidden_units'] = 512
  flow_params['is_iaf'] = False
  flow_bijector_pre = build_iaf_bijector(**flow_params)
  flow_bijector_post = build_iaf_bijector(**flow_params)
  make_swap = lambda: tfb.Permute(ps.range(ndims - 1, -1, -1))
  if use_gates:
    structured_layer = StructuredLayer(prior)
  else:
    structured_layer = GatedStructuredLayer(prior)
  prior_matching_bijectors = tfb.Chain(prior_matching_bijectors)

  if use_bn:
    bijector = tfb.Chain([prior_matching_bijectors,
                          flow_bijector_post[0],
                          tfb.Invert(ActivationNormalization(1,
                                                             is_image=False)),
                          tfb.Chain([tfb.Invert(prior_matching_bijectors),
                                     structured_layer,
                                     prior_matching_bijectors]),
                          make_swap(),
                          tfb.Invert(ActivationNormalization(1,
                                                             is_image=False)),
                          flow_bijector_pre[0],
                          tfb.Invert(ActivationNormalization(1,
                                                             is_image=False))
                          ])
  else:
    bijector = tfb.Chain([prior_matching_bijectors,
                          flow_bijector_post[0],
                          tfb.Chain([tfb.Invert(prior_matching_bijectors),
                                     structured_layer,
                                     prior_matching_bijectors]),
                          make_swap(),
                          flow_bijector_pre[0]
                          ])

  backbone_surrogate_posterior = tfd.TransformedDistribution(
    distribution=base_distribution,
    bijector=bijector
  )

  return backbone_surrogate_posterior


def nsf_emf_middle(prior, flow_params, use_gates=False):
  event_shape, flat_event_shape, flat_event_size, ndims, dtype, prior_matching_bijectors = get_prior_matching_bijectors_and_event_dims(
    prior)

  base_distribution = tfd.Sample(
    tfd.Normal(tf.zeros([], dtype=dtype), 1.), sample_shape=[ndims])

  flow_bijector_pre = make_splines(**flow_params)
  flow_bijector_post = make_splines(**flow_params)
  make_swap = lambda: tfb.Permute(ps.range(ndims - 1, -1, -1))
  if use_gates:
    structured_layer = StructuredLayer(prior)
  else:
    structured_layer = GatedStructuredLayer(prior)
  prior_matching_bijectors = tfb.Chain(prior_matching_bijectors)

  if flow_params['use_bn']:
    bijector = tfb.Chain([prior_matching_bijectors,
                          tfb.Chain(flow_bijector_post),
                          tfb.Invert(ActivationNormalization(1,
                                                             is_image=False)),
                          tfb.Chain([tfb.Invert(prior_matching_bijectors),
                                     structured_layer,
                                     prior_matching_bijectors]),
                          make_swap(),
                          tfb.Invert(ActivationNormalization(1,
                                                             is_image=False)),
                          tfb.Chain(flow_bijector_pre)])
  else:
    bijector = tfb.Chain([prior_matching_bijectors,
                          tfb.Chain(flow_bijector_post),
                          tfb.Chain([tfb.Invert(prior_matching_bijectors),
                                     structured_layer,
                                     prior_matching_bijectors]),
                          make_swap(),
                          tfb.Chain(flow_bijector_pre)])

  backbone_surrogate_posterior = tfd.TransformedDistribution(
    distribution=base_distribution,
    bijector=bijector
  )

  return backbone_surrogate_posterior


def emf_bottom(prior, flow_params={}, use_gates=True):
  event_shape, flat_event_shape, flat_event_size, ndims, dtype, prior_matching_bijectors = get_prior_matching_bijectors_and_event_dims(
    prior)

  base_distribution = tfd.Sample(
    tfd.Normal(tf.zeros([], dtype=dtype), 1.), sample_shape=[ndims])

  flow_params['activation_fn'] = tf.nn.relu
  flow_params['dtype'] = dtype
  flow_params['ndims'] = ndims
  flow_params['num_flow_layers'] = 1
  if 'num_hidden_units' not in flow_params:
    flow_params['num_hidden_units'] = 512
  flow_params['is_iaf'] = False
  flow_bijector_pre = build_iaf_bijector(**flow_params)
  flow_bijector_post = build_iaf_bijector(**flow_params)
  if use_gates:
    structured_layer = StructuredLayer(prior)
  else:
    structured_layer = GatedStructuredLayer(prior)
  prior_matching_bijectors = tfb.Chain(prior_matching_bijectors)

  bijector = tfb.Chain([
    prior_matching_bijectors,
    flow_bijector_post[0],
    flow_bijector_pre[0],
    tfb.Chain([tfb.Invert(prior_matching_bijectors),
               structured_layer,
               prior_matching_bijectors])
  ])

  backbone_surrogate_posterior = tfd.TransformedDistribution(
    distribution=base_distribution,
    bijector=bijector
  )

  return backbone_surrogate_posterior
