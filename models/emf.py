import tensorflow_probability as tfp

from bijectors.gated_structured_layer import GatedStructuredLayer
from bijectors.structured_layer import StructuredLayer

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental
tfp_util = tfp.util


def embedded_model_flow(prior, backbone_model):
  bijector = StructuredLayer(prior)

  return tfd.TransformedDistribution(
    distribution=backbone_model,
    bijector=bijector
  )


def gated_emf(prior, backbone_model):
  bijector = GatedStructuredLayer(prior)
  return tfd.TransformedDistribution(
    distribution=backbone_model,
    bijector=bijector
  )
