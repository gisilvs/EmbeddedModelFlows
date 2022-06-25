import tensorflow as tf
import tensorflow_probability as tfp

from bijectors.gate_bijector import GateBijector, GateBijectorForNormal
from bijectors.mixture_of_gaussian_bijector import InverseMixtureOfGaussians

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


gated_stdnormal_bijector_fns = {
  tfd.Gamma: lambda d: tfd.ApproxGammaFromNormal(d.concentration,
                                                 d._rate_parameter()),
  # using specific bijector for normal, use next line for generic one
  tfd.Normal: lambda d: GateBijectorForNormal(d.loc, d.scale,
                                              get_residual_fraction(d)),
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
  tfd.Independent: lambda d: _gated_bijector_from_stdnormal(d.distribution),
  tfd.MixtureSameFamily: lambda d: GateBijector(
    tfb.Chain([InverseMixtureOfGaussians(d), tfb.NormalCDF()]),
    get_residual_fraction(d))
}

gated_stdnormal_bijector_sample_fns = {
  tfd.Normal: lambda d: GateBijector(tfb.Shift(tf.reshape(d.loc, [-1, 1]))(
    tfb.Scale(tf.reshape(d.scale, [-1, 1]))), get_residual_fraction(d))
}


def _gated_bijector_from_stdnormal(dist):
  fn = gated_stdnormal_bijector_fns[type(dist)]
  return fn(dist)


def _gated_bijector_from_stdnormal_sample(dist):
  fn = gated_stdnormal_bijector_sample_fns[type(dist)]
  return fn(dist)


class GatedStructuredLayer(tfd.joint_distribution._DefaultJointBijector):

  def __init__(self, dist):
    return super().__init__(dist, bijector_fn=_gated_bijector_from_stdnormal)
