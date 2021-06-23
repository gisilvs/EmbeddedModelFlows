import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import pickle

import matplotlib.pyplot as plt

import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors
Root = tfd.JointDistributionCoroutine.Root

def _brownian_motion(is_bridge, is_classification, seed=None):
  @tfd.JointDistributionCoroutineAutoBatched
  def model():
    innovation_noise= .1
    observation_noise = .15
    k = 5.
    truth = []
    new = yield Root(tfd.Normal(loc=0.,
                                scale=innovation_noise,
                                name='x_0'))
    truth.append(new)

    for t in range(1, 30):
      new = yield tfd.Normal(loc=new,
                             scale=innovation_noise,
                             name=f'x_{t}')
      truth.append(new)
    if is_bridge:
      time_steps = list(range(10)) + list(range(20,30))
    else:
      time_steps = range(30)
    for t in time_steps:
      if is_classification:
        yield tfd.Bernoulli(logits=k * truth[t], name=f'y_{t}')
      else:
        yield tfd.Normal(loc=truth[t],
                         scale=observation_noise,
                         name=f'y_{t}')

  ground_truth = model.sample(seed=seed)
  brownian_bridge = model.experimental_pin(ground_truth[30:])

  return brownian_bridge, ground_truth[:30], brownian_bridge.unnormalized_log_prob, ground_truth[30:]

def _lorenz_system(is_bridge, is_classification, seed=None):
  @tfd.JointDistributionCoroutineAutoBatched
  def model():
    truth = []
    innovation_noise = .1
    observation_noise = 1.
    k = 2.
    step_size = 0.02
    loc = yield Root(tfd.Sample(tfd.Normal(0., 1.), sample_shape=3, name='x_0'))
    for t in range(1, 30):
      x, y, z = tf.unstack(loc, axis=-1)
      truth.append(x)
      dx = 10 * (y - x)
      dy = x * (28 - z) - y
      dz = x * y - 8 / 3 * z
      delta = tf.stack([dx, dy, dz], axis=-1)
      loc = yield tfd.Independent(
        tfd.Normal(loc + step_size * delta,
                   tf.sqrt(step_size) * innovation_noise),
        reinterpreted_batch_ndims=1, name=f'x_{t}')
    x, y, z = tf.unstack(loc, axis=-1)
    truth.append(x)

    if is_bridge:
      time_steps = list(range(10)) + list(range(20, 30))
    else:
      time_steps = range(30)
    for t in time_steps:
      if is_classification:
        yield tfd.Bernoulli(logits=k * truth[t], name=f'y_{t}')
      else:
        yield tfd.Normal(loc=truth[t],
                         scale=observation_noise,
                         name=f'y_{t}')

  ground_truth = model.sample(seed=seed)
  lorenz_bridge = model.experimental_pin(ground_truth[30:])
  return lorenz_bridge, ground_truth[:30], lorenz_bridge.unnormalized_log_prob, ground_truth[30:]


def _eight_schools(seed=None):
  num_schools = 8  # number of schools
  treatment_effects = np.array(
    [28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)  # treatment effects
  treatment_stddevs = np.array(
    [15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)  # treatment SE

  model = tfd.JointDistributionSequential([
    tfd.Normal(loc=0., scale=10., name="avg_effect"),  # `mu` above
    tfd.Normal(loc=5., scale=1., name="avg_stddev"),  # `log(tau)` above
    tfd.Independent(tfd.Normal(loc=tf.zeros(num_schools),
                               scale=tf.ones(num_schools),
                               name="school_effects_standard"),
                    # `theta_prime`
                    reinterpreted_batch_ndims=1),
    lambda school_effects_standard, avg_stddev, avg_effect: (
      tfd.Independent(tfd.Normal(loc=(avg_effect[..., tf.newaxis] +
                                      tf.exp(avg_stddev[..., tf.newaxis]) *
                                      school_effects_standard),  # `theta` above
                                 scale=treatment_stddevs),
                      name="treatment_effects",  # `y` above
                      reinterpreted_batch_ndims=1))
  ])

  observations = {'treatment_effects': treatment_effects,
                  'treatment_stddevs': treatment_stddevs}

  eight_schools = model.experimental_pin(treatment_effects=treatment_effects)

  # if seed is one of those used in run_experiments.py, then we can use them as index of ground truth
  idx = int(seed / 10 -1)
  with open(f'ground_truth/eight_schools/gt.pickle', 'rb') as handle:
    gt = pickle.load(handle)

  return eight_schools, gt[idx], eight_schools.unnormalized_log_prob, observations

def _radon(seed):

  dataset = tfds.as_numpy(
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

  # Define the probabilistic graphical model as a JointDistribution.
  @tfd.JointDistributionCoroutineAutoBatched
  def model():
    county_effect_mean = yield tfd.Normal(0., 1., name='county_effect_mean')
    county_effect_scale = yield tfd.HalfNormal(scale=1., name='county_effect_scale')

    county_effect = yield tfd.Sample(
      tfd.Normal(county_effect_mean, scale=county_effect_scale),
      sample_shape=[num_counties], name='county_effect')

    uranium_weight = yield tfd.Normal(0., scale=1., name='uranium_weight')
    county_floor_weight = yield tfd.Normal(
      0., scale=1., name='county_floor_weight')
    floor_weight = yield tfd.Normal(0., scale=1., name='floor_weight')

    log_radon_scale = yield tfd.HalfNormal(scale=1., name='log_radon_scale')
    yield tfd.Normal(
      loc=(log_uranium * uranium_weight + floor_of_house * floor_weight
           + floor_by_county * county_floor_weight
           + tf.gather(county_effect, county, axis=-1)),
      scale=log_radon_scale[..., tf.newaxis],
      name='log_radon')

    # Pin the observed `log_radon` values to model the un-normalized posterior.

  target_model = model.experimental_pin(log_radon=log_radon)

  idx = int(seed / 10 - 1)
  with open(f'ground_truth/radon/gt.pickle', 'rb') as handle:
    gt = pickle.load(handle)

  return target_model, gt[idx], target_model.unnormalized_log_prob, log_radon

def _gaussian_binary_tree(num_layers, initial_scale, nodes_scale, coupling_link, seed=None):
  @tfd.JointDistributionCoroutineAutoBatched
  def collider_model():
    layers = yield Root(tfd.Sample(tfd.Normal(0., initial_scale), 2 ** (num_layers - 1), name=f'layer_{num_layers - 1}'))
    for l in range((num_layers - 1), 0, -1):
      if coupling_link:
        layers = yield tfd.Independent(tfd.Normal(tf.stack(
          [coupling_link(layers[..., i]) - coupling_link(layers[..., i + 1]) for i in range(0, 2 ** l, 2)],
          -1),
          nodes_scale), 1, name=f'layer_{l - 1}')
      else:
        layers = yield tfd.Independent(tfd.Normal(tf.stack(
          [layers[..., i] - layers[..., i + 1] for i in range(0, 2 ** l, 2)], -1),
                                                  nodes_scale), 1, name=f'layer_{l-1}')

  ground_truth = collider_model.sample(seed=seed)
  model = collider_model.experimental_pin(layer_0=ground_truth[-1])

  return model, ground_truth[:-1], model.unnormalized_log_prob, ground_truth[-1]

def get_model(model_name, seed=None):

  if model_name=='brownian_smoothing_r':
    return _brownian_motion(is_bridge=False, is_classification=False, seed=seed)

  elif model_name=='brownian_smoothing_c':
    return _brownian_motion(is_bridge=False, is_classification=True, seed=seed)

  if model_name=='brownian_bridge_r':
    return _brownian_motion(is_bridge=True, is_classification=False, seed=seed)

  elif model_name=='brownian_bridge_c':
    return _brownian_motion(is_bridge=True, is_classification=True, seed=seed)

  elif model_name=='lorenz_smoothing_r':
    return _lorenz_system(is_bridge=False, is_classification=False, seed=seed)

  elif model_name=='lorenz_smoothing_c':
    return _lorenz_system(is_bridge=False, is_classification=True, seed=seed)

  elif model_name=='lorenz_bridge_r':
    return _lorenz_system(is_bridge=True, is_classification=False, seed=seed)

  elif model_name=='lorenz_bridge_c':
    return _lorenz_system(is_bridge=True, is_classification=True, seed=seed)

  elif model_name=='eight_schools':
    return _eight_schools(seed=seed)

  elif model_name=='radon':
    return _radon(seed=seed)

  elif model_name=='linear_binary_tree_4':
    return _gaussian_binary_tree(num_layers=4, initial_scale=0.2, nodes_scale=0.15, coupling_link=None, seed=seed)

  elif model_name=='linear_binary_tree_8':
    return _gaussian_binary_tree(num_layers=8, initial_scale=0.2, nodes_scale=0.15, coupling_link=None, seed=seed)

  elif model_name=='tanh_binary_tree_4':
    return _gaussian_binary_tree(num_layers=4, initial_scale=0.1, nodes_scale=0.05, coupling_link=tf.nn.tanh, seed=seed)

  elif model_name=='tanh_binary_tree_8':
    return _gaussian_binary_tree(num_layers=8, initial_scale=0.1, nodes_scale=0.05, coupling_link=tf.nn.tanh, seed=seed)
