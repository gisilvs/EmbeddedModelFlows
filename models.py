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
  def target_log_prob_fn(avg_effect, log_stddev, school_effects):
    samples_as_dict = {'avg_effect': avg_effect,
       'log_stddev': log_stddev,
       'school_effects': school_effects
       }
    return model.log_likelihood(samples_as_dict) + prior.log_prob(samples_as_dict)
  #target_log_prob = lambda *values: prior.log_prob(values)

  treatment_effects = tf.constant(
    [28, 8, -3, 7, -1, 1, 18, 12], dtype=tf.float32)
  treatment_stddevs = tf.constant(
    [15, 10, 16, 11, 9, 11, 10, 18], dtype=tf.float32)

  observations = {'treatment_effects': treatment_effects,
                  'treatment_stddevs': treatment_stddevs}

  return model, prior, ground_truth, target_log_prob_fn, observations

def get_model(model_name):
  if model_name=='brownian_bridge':
    return _brownian_bridge()

  elif model_name=='lorenz_bridge':
    return _lorenz_bridge()

  elif model_name=='eight_schools':
    return _eight_schools()
