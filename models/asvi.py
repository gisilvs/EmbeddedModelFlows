import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental
tfp_util = tfp.util


def asvi(prior):
  return tfe.vi.build_asvi_surrogate_posterior(prior)
