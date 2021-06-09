import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.bijectors import build_trainable_highway_flow, HighwayFlow
from tensorflow_probability.python.internal import samplers

tfb = tfp.bijectors

def build_trainable_thighway_flow_without_gating(width, activation_fn,
                             seed=None):
  residual_fraction_initial_value = tf.convert_to_tensor(
    0.,
    dtype_hint=tf.float32,
    name='residual_fraction_initial_value')
  dtype =tf.float32

  bias_seed, upper_seed, lower_seed = samplers.split_seed(seed, n=3)
  lower_bijector = tfb.Chain([
    tfb.TransformDiagonal(diag_bijector=tfb.Shift(1.)),
    tfb.Pad(paddings=[(1, 0), (0, 1)]),
    tfb.FillTriangular()
  ])
  unconstrained_lower_initial_values = samplers.normal(
    shape=lower_bijector.inverse_event_shape([width, width]),
    mean=0.,
    stddev=.01,
    seed=lower_seed)
  upper_bijector = tfb.FillScaleTriL(
    diag_bijector=tfb.Softplus(), diag_shift=None)
  unconstrained_upper_initial_values = samplers.normal(
    shape=upper_bijector.inverse_event_shape([width, width]),
    mean=0.,
    stddev=.01,
    seed=upper_seed)

  return HighwayFlow(
    residual_fraction=tfp.util.TransformedVariable(
      initial_value=residual_fraction_initial_value,
      bijector=tfb.Sigmoid(),
      dtype=dtype,
      trainable=False),
    activation_fn=activation_fn,
    bias=tf.Variable(
      samplers.normal((width,), mean=0., stddev=0.01, seed=bias_seed),
      dtype=dtype),
    upper_diagonal_weights_matrix=tfp.util.TransformedVariable(
      initial_value=upper_bijector.forward(
        unconstrained_upper_initial_values),
      bijector=upper_bijector,
      dtype=dtype),
    lower_diagonal_weights_matrix=tfp.util.TransformedVariable(
      initial_value=lower_bijector.forward(
        unconstrained_lower_initial_values),
      bijector=lower_bijector,
      dtype=dtype),
    gate_first_n=0)

def build_highway_flow_bijector_without_gating(num_layers, width, seed=None):
  bijectors = []

  for _ in range(0, num_layers - 1):
    bijectors.append(
      build_trainable_thighway_flow_without_gating(width,
                                   activation_fn=tf.nn.softplus,seed=seed))
  bijectors.append(
    build_trainable_thighway_flow_without_gating(width,
                                 activation_fn=None,seed=seed))

  return bijectors

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