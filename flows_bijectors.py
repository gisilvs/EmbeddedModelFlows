import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.bijectors import build_trainable_highway_flow

tfb = tfp.bijectors

def build_highway_flow_bijector(num_layers, width,
                             residual_fraction_initial_value, gate_first_n,
                             seed=None):
  bijectors = []

  for _ in range(0, num_layers - 1):
    bijectors.append(
      build_trainable_highway_flow(width,
                                   residual_fraction_initial_value=residual_fraction_initial_value,
                                   activation_fn=tf.nn.softplus,
                                   gate_first_n=gate_first_n, seed=seed))
  bijectors.append(
    build_trainable_highway_flow(width,
                                 residual_fraction_initial_value=residual_fraction_initial_value,
                                 activation_fn=None,
                                 gate_first_n=gate_first_n, seed=seed))

  return bijectors

def build_iaf_biector(num_iafs, hidden_units):
  # todo: should params always be 2?
  iaf_bijectors = [
    tfb.Invert(tfb.MaskedAutoregressiveFlow(
      shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
        params=2, hidden_units=[hidden_units, hidden_units],
        activation='relu')))  # todo: do we need relu here?
    for _ in range(num_iafs)
  ]

  return iaf_bijectors