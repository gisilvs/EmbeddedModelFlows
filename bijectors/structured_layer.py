import tensorflow as tf
import tensorflow_probability as tfp

from bijectors.mixture_of_gaussian_bijector import InverseMixtureOfGaussians

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental
tfp_util = tfp.util

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
  tfd.MixtureSameFamily: lambda d: tfb.Chain(
    [InverseMixtureOfGaussians(d), tfb.NormalCDF()])
}

stdnormal_bijector_sample_fns = {
  tfd.Normal: lambda d: tfb.Shift(tf.reshape(d.loc, [-1, 1]))(
    tfb.Scale(tf.reshape(d.scale, [-1, 1])))
}


def _bijector_from_stdnormal_sample(dist):
  fn = stdnormal_bijector_sample_fns[type(dist)]
  return fn(dist)


def _bijector_from_stdnormal(dist):
  fn = stdnormal_bijector_fns[type(dist)]
  return fn(dist)


class StructuredLayer(tfd.joint_distribution._DefaultJointBijector):

  def __init__(self, dist):
    return super().__init__(dist, bijector_fn=_bijector_from_stdnormal)
