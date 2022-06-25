import tensorflow as tf
import tensorflow_probability as tfp

from models.emf_middle import emf_middle, nsf_emf_middle, emf_bottom
from models.model_getter import get_model
from utils.utils import get_prior_matching_bijectors_and_event_dims

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root


def get_hierarchical_prior(dataset_name):
  if dataset_name == 'iris':
    scales = tf.ones(4)
    initial_mean = tf.zeros(4)
    length = 10
  elif dataset_name == 'digits':
    scales = tf.ones(64)
    initial_mean = tf.zeros(64)
    length = 20

  @tfd.JointDistributionCoroutine
  def prior_structure():
    mean = yield Root(tfd.Independent(tfd.Normal(loc=initial_mean,
                                                 scale=scales,
                                                 name='prior0'), 1))

    for t in range(1, length):
      new = yield tfd.Independent(tfd.Normal(loc=mean,
                                             scale=scales, name=f'prior{t}'),
                                  1)

  return prior_structure


def get_timeseries_prior(model_name, prior_name, time_step_dim):
  if model_name == 'maf' or model_name == 'maf3' or model_name == \
      'splines':
    scales = tf.ones(time_step_dim)
  else:
    scales = tfp.util.TransformedVariable(tf.ones(time_step_dim),
                                          tfb.Softplus())
  initial_mean = tf.zeros(time_step_dim)

  if prior_name == 'continuity':
    @tfd.JointDistributionCoroutine
    def prior_structure():
      new = yield Root(tfd.Independent(tfd.Normal(loc=initial_mean,
                                                  scale=tf.ones_like(
                                                    initial_mean),
                                                  name='prior0'), 1))

      for t in range(1, time_step_dim):
        new = yield tfd.Independent(tfd.Normal(loc=new,
                                               scale=scales,
                                               name=f'prior{t}'), 1)

  elif prior_name == 'smoothness':
    @tfd.JointDistributionCoroutine
    def prior_structure():
      previous = yield Root(tfd.Independent(tfd.Normal(loc=initial_mean,
                                                       scale=tf.ones_like(
                                                         initial_mean),
                                                       name='prior0'), 1))
      current = yield Root(tfd.Independent(tfd.Normal(loc=initial_mean,
                                                      scale=tf.ones_like(
                                                        initial_mean),
                                                      name='prior1'), 1))
      for t in range(2, time_step_dim):
        new = yield tfd.Independent(tfd.Normal(loc=2 * current - previous,
                                               scale=scales,
                                               name=f'prior{t}'), 1)
        previous = current
        current = new

  return prior_structure


def get_mixture_prior(model_name, n_components, n_dims, trainable_mixture=True,
                      component_logits=None,
                      locs=None, scales=None):
  if trainable_mixture:
    if model_name == 'maf' or model_name == 'maf3' \
        or model_name == 'splines':
      component_logits = tf.convert_to_tensor(
        [[1. / n_components for _ in range(n_components)] for _ in
         range(n_dims)])
      locs = tf.convert_to_tensor(
        [tf.linspace(-n_components / 2, n_components / 2, n_components) for _
         in
         range(n_dims)])
      scales = tf.convert_to_tensor(
        [[1. for _ in range(n_components)] for _ in
         range(n_dims)])
    else:
      if model_name == 'embedded_model_flow':
        loc_range = 4.
        scale = 1.
      else:
        loc_range = 10.
        scale = 3.
      component_logits = tf.Variable(
        [[1. / n_components for _ in range(n_components)] for _ in
         range(n_dims)], name='component_logits')
      locs = tf.Variable(
        [tf.linspace(-loc_range, loc_range, n_components) for _ in
         range(n_dims)],
        name='locs')
      scales = tfp.util.TransformedVariable(
        [[scale for _ in range(n_components)] for _ in
         range(n_dims)], tfb.Softplus(), name='scales')

  @tfd.JointDistributionCoroutine
  def prior_structure():
    yield Root(tfd.Independent(tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(logits=component_logits),
      components_distribution=tfd.Normal(loc=locs, scale=scales),
      name=f"prior"), 1))

  return prior_structure


def build_model(model_name, prior_structure, flow_dim):
  prior_matching_bijector = tfb.Chain(
    get_prior_matching_bijectors_and_event_dims(
      prior_structure)[-1])
  if model_name == 'maf':
    model = get_model(prior_structure, 'maf')

  elif model_name == 'maf3':
    flow_params = {'num_flow_layers': 3}
    model = get_model(prior_structure, 'maf', flow_params=flow_params)
  elif model_name == 'embedded_model_flow':
    model = get_model(prior_structure,
                      'embedded_model_flow',
                      'maf')
  elif model_name == 'emf_middle':
    model = emf_middle(prior_structure)

  elif model_name == 'nsf_emf_middle':
    flow_params = {
      'layers': 3,
      'number_of_bins': 32,
      'input_dim': flow_dim,
      'nn_layers': [32, 32],
      'b_interval': 4,
      'use_bn': False
    }
    model = nsf_emf_middle(
      prior_structure, flow_params)

  elif model_name == 'splines':
    flow_params = {
      'layers': 6,
      'number_of_bins': 32,
      'input_dim': flow_dim,
      'nn_layers': [32, 32],
      'b_interval': 4,
      'use_bn': False
    }
    model = get_model(prior_structure, model_name='splines',
                      flow_params=flow_params)

  elif model_name == 'nsf_emf_top':
    flow_params = {
      'layers': 6,
      'number_of_bins': 32,
      'input_dim': flow_dim,
      'nn_layers': [32, 32],
      'b_interval': 4,
      'use_bn': False
    }
    model = get_model(prior_structure,
                      model_name='embedded_model_flow',
                      backnone_name='splines',
                      flow_params=flow_params)
    # maf.sample(1)

  elif model_name == 'maf_b':
    model = emf_bottom(prior_structure, {}, use_gates=False)

  elif model_name == 'emf_m':
    model = emf_middle(prior_structure, {})

  elif model_name == 'nsf_emf_m':
    model = nsf_emf_middle(prior_structure, {})

  model.log_prob(prior_structure.sample(2))

  return model, prior_matching_bijector
