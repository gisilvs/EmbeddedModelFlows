import os
import shutil

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental
tfp_util = tfp.util


def get_prior_matching_bijectors_and_event_dims(prior):
  event_shape = prior.event_shape_tensor()
  flat_event_shape = tf.nest.flatten(event_shape)
  flat_event_size = tf.nest.map_structure(tf.reduce_prod, flat_event_shape)
  try:
    event_space_bijector = prior.experimental_default_event_space_bijector()
  except:
    event_space_bijector = None

  split_bijector = tfb.Split(flat_event_size)
  unflatten_bijector = tfb.Restructure(
    tf.nest.pack_sequence_as(
      event_shape, range(len(flat_event_shape))))
  reshape_bijector = tfb.JointMap(
    tf.nest.map_structure(tfb.Reshape, flat_event_shape,
                          [x[tf.newaxis] for x in flat_event_size]))

  if event_space_bijector:

    prior_matching_bijectors = [event_space_bijector, unflatten_bijector,
                                reshape_bijector, split_bijector]

  else:
    prior_matching_bijectors = [unflatten_bijector,
                                reshape_bijector, split_bijector]

  dtype = tf.nest.flatten(prior.dtype)[0]

  return event_shape, flat_event_shape, flat_event_size, int(
    tf.reduce_sum(flat_event_size)), dtype, prior_matching_bijectors


def clear_folder(folder):
  for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
      elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
    except Exception as e:
      print('Failed to delete %s. Reason: %s' % (file_path, e))
