import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from inference_gym import using_tensorflow as gym
from inference_gym.internal.datasets import brownian_motion_missing_middle_observations, convection_lorenz_bridge

import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors
Root = tfd.JointDistributionCoroutine.Root

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

  '''dataset = tfds.as_numpy(
    tfds.load('radon', split='train').filter(
      lambda x: x['features']['state'] == 'MN').batch(10 ** 9))

  # Dependent variable: Radon measurements by house.
  dataset = next(iter(dataset))
  radon_measurement = dataset['activity'].astype(np.float32)
  radon_measurement[radon_measurement <= 0.] = 0.1
  log_radon = np.log(radon_measurement)

  # Measured uranium concentrations in surrounding soil.
  uranium_measurement = dataset['features']['Uppm'].astype(np.float32)
  log_uranium = np.log(uranium_measurement)

  # County indicator.
  county_strings = dataset['features']['county'].astype('U13')
  unique_counties, county = np.unique(county_strings, return_inverse=True)
  county = county.astype(np.int32)
  num_counties = unique_counties.size

  # Floor on which the measurement was taken.
  floor_of_house = dataset['features']['floor'].astype(np.int32)

  # Average floor by county (contextual effect).
  county_mean_floor = []
  for i in range(num_counties):
    county_mean_floor.append(floor_of_house[county == i].mean())
  county_mean_floor = np.array(county_mean_floor, dtype=log_radon.dtype)
  floor_by_county = county_mean_floor[county]

  # Create variables for fixed effects.
  floor_weight = tf.Variable(0.)
  bias = tf.Variable(0.)

  # Variables for scale parameters.
  log_radon_scale = tfp.util.TransformedVariable(1., tfb.Exp())
  county_effect_scale = tfp.util.TransformedVariable(1., tfb.Exp())

  # Define the probabilistic graphical model as a JointDistribution.
  @tfd.JointDistributionCoroutineAutoBatched
  def model():
    uranium_weight = yield tfd.Normal(0., scale=1., name='uranium_weight')
    county_floor_weight = yield tfd.Normal(
      0., scale=1., name='county_floor_weight')
    county_effect = yield tfd.Sample(
      tfd.Normal(0., scale=county_effect_scale),
      sample_shape=[num_counties], name='county_effect')
    yield tfd.Normal(
      loc=(log_uranium * uranium_weight + floor_of_house * floor_weight

           + floor_by_county * county_floor_weight
           + tf.gather(county_effect, county, axis=-1)
           + bias),
      scale=log_radon_scale[..., tf.newaxis],
      name='log_radon')

    # Pin the observed `log_radon` values to model the un-normalized posterior.

  target_model = model.experimental_pin(log_radon=log_radon)'''
  # todo: is minnesota the one used in asvi paper?
  model = gym.targets.RadonContextualEffectsMinnesota(dtype=tf.float32)
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

def _gaussian_binary_tree(num_layers, initial_scale, nodes_scale, coupling_link):
  @tfd.JointDistributionCoroutineAutoBatched
  def collider_model():
    layers = yield Root(tfd.Sample(tfd.Normal(0., initial_scale), 2 ** num_layers, name=f'layer_{num_layers}'))
    for l in range(num_layers, 0, -1):
      if coupling_link:
        layers = yield tfd.Independent(tfd.Normal(tf.stack(
          [coupling_link(layers[..., i]) - coupling_link(layers[..., i + 1]) for i in range(0, 2 ** l, 2)],
          -1),
          nodes_scale), 1, name=f'layer_{l - 1}')
      else:
        layers = yield tfd.Independent(tfd.Normal(tf.stack(
          [layers[..., i] - layers[..., i + 1] for i in range(0, 2 ** l, 2)], -1),
                                                  nodes_scale), 1, name=f'layer_{l-1}')

  ground_truth = collider_model.sample(seed=44)
  model = collider_model.experimental_pin(layer_0=ground_truth[-1])

  return None, model, ground_truth[:-1], model.unnormalized_log_prob, ground_truth[-1]

def get_model(model_name):
  if model_name=='brownian_bridge':
    return _brownian_bridge()

  elif model_name=='lorenz_bridge':
    return _lorenz_bridge()

  elif model_name=='eight_schools':
    return _eight_schools()

  elif model_name=='radon':
    return _radon()

  elif model_name=='linear_binary_tree_small':
    return _gaussian_binary_tree(num_layers=2, initial_scale=0.2, nodes_scale=0.15, coupling_link=None)

  elif model_name=='linear_binary_tree_large':
    return _gaussian_binary_tree(num_layers=4, initial_scale=0.2, nodes_scale=0.15, coupling_link=None)

  elif model_name=='tanh_binary_tree_small':
    return _gaussian_binary_tree(num_layers=2, initial_scale=0.1, nodes_scale=0.05, coupling_link=tf.nn.tanh)

  elif model_name=='tanh_binary_tree_large':
    return _gaussian_binary_tree(num_layers=4, initial_scale=0.1, nodes_scale=0.05, coupling_link=tf.nn.tanh)
