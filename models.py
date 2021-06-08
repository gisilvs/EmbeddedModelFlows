import tensorflow as tf
from inference_gym import using_tensorflow as gym
from inference_gym.internal.datasets import brownian_motion_missing_middle_observations, convection_lorenz_bridge

def _brownian_bridge():
  model = gym.targets.BrownianMotionMissingMiddleObservations()
  prior = model.prior_distribution()
  ground_truth = model.sample_transformations['identity'].ground_truth_mean
  target_log_prob = lambda *values: model.log_likelihood(values) + \
                                    prior.log_prob(values)
  OBSERVED_LOC = brownian_motion_missing_middle_observations.OBSERVED_LOC

  return model, prior, ground_truth, target_log_prob, OBSERVED_LOC

def _lorenz_bridge():
  model = gym.targets.ConvectionLorenzBridge()
  prior = model.prior_distribution()
  ground_truth = model.sample_transformations['identity'].ground_truth_mean
  target_log_prob = lambda *values: model.log_likelihood(values) + \
                                    prior.log_prob(values)
  OBSERVED_VALUES = convection_lorenz_bridge.OBSERVED_VALUES

  return model, prior, ground_truth, target_log_prob, OBSERVED_VALUES

def _eight_schools():

  model = gym.targets.EightSchools()
  prior = model.prior_distribution()
  ground_truth = model.sample_transformations['identity'].ground_truth_mean
  def target_log_prob(avg_effect, log_stddev, school_effects):
    samples_as_dict = {'avg_effect': avg_effect,
       'log_stddev': log_stddev,
       'school_effects': school_effects
       }
    return model.log_likelihood(samples_as_dict) + prior.log_prob(samples_as_dict)

  treatment_effects = tf.constant(
    [28, 8, -3, 7, -1, 1, 18, 12], dtype=tf.float32)
  treatment_stddevs = tf.constant(
    [15, 10, 16, 11, 9, 11, 10, 18], dtype=tf.float32)

  observations = {'treatment_effects': treatment_effects,
                  'treatment_stddevs': treatment_stddevs}

  return model, prior, ground_truth, target_log_prob, observations

def _radon():

  # todo: is minnesota the one used in asvi paper?
  model = gym.targets.RadonContextualEffectsMinnesota()
  prior = model.prior_distribution()
  ground_truth = model.sample_transformations['identity'].ground_truth_mean
  def target_log_prob(county_effect_mean, county_effect_scale, county_effect, weight, log_radon_scale):
    samples_as_dict = {'county_effect_mean': county_effect_mean,
                       'county_effect_scale': county_effect_scale,
                       'county_effect': county_effect,
                       'weight': weight,
                       'log_radon_scale': log_radon_scale
       }
    return model.log_likelihood(samples_as_dict) + prior.log_prob(samples_as_dict)
  # todo: do we have observations in this model?
  return model, prior, ground_truth, target_log_prob, None


def get_model(model_name):
  if model_name=='brownian_bridge':
    return _brownian_bridge()

  elif model_name=='lorenz_bridge':
    return _lorenz_bridge()

  elif model_name=='eight_schools':
    return _eight_schools()

  elif model_name=='radon':
    return _radon()
