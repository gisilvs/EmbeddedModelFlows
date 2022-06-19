import tensorflow as tf
import tensorflow_probability as tfp

from models.asvi import asvi
from models.emf import embedded_model_flow, gated_emf
from models.mean_field import mean_field
from models.multivariate_normal import multivariate_normal
from models.normalizing_flows import normalizing_flows

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental
tfp_util = tfp.util


def get_model(prior, model_name, backnone_name=None, flow_params={}):
  # Needed to reset the gates if running several experiments sequentially
  global residual_fraction_vars
  residual_fraction_vars = {}

  if model_name == 'mean_field':
    return mean_field(prior)

  elif model_name == 'multivariate_normal':
    return multivariate_normal(prior)

  elif model_name == "asvi":
    return asvi(prior)

  elif model_name == "iaf":
    flow_params['num_flow_layers'] = 2
    flow_params['num_hidden_units'] = 512
    if 'activation_fn' not in flow_params:
      flow_params['activation_fn'] = tf.math.tanh
    return normalizing_flows(prior, flow_name='iaf', flow_params=flow_params)

  elif model_name == "maf":
    if 'num_flow_layers' not in flow_params:
      flow_params['num_flow_layers'] = 2
    if 'num_hidden_units' not in flow_params:
      flow_params['num_hidden_units'] = 512
    flow_params['is_iaf'] = False
    if 'activation_fn' not in flow_params:
      flow_params['activation_fn'] = tf.math.tanh
    return normalizing_flows(prior, flow_name='maf', flow_params=flow_params)

  elif model_name == "splines":
    return normalizing_flows(prior, flow_name='splines',
                             flow_params=flow_params)

  elif model_name == "embedded_model_flow":
    if backnone_name == 'iaf':
      flow_params['activation_fn'] = tf.nn.relu
    elif backnone_name == 'maf':
      flow_params['activation_fn'] = tf.nn.relu

    model = get_model(prior, model_name=backnone_name, flow_params=flow_params)
    return embedded_model_flow(prior=prior, backbone_model=model)

  elif model_name == "gated_emf":
    if backnone_name == 'iaf':
      flow_params['activation_fn'] = tf.nn.relu
    elif backnone_name == 'maf':
      flow_params['activation_fn'] = tf.nn.relu

    model = get_model(prior, model_name=backnone_name, flow_params=flow_params)
    return gated_emf(prior=prior, backbone_model=model)
