import random
import pickle

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from models import get_model

tfb = tfp.bijectors

target_model , _, target_log_prob_fn, _ =get_model('radon', seed=10)

num_chains = 8
num_leapfrog_steps = 3
step_size = 0.4
num_steps=20000

flat_event_shape = tf.nest.flatten(target_model.event_shape)
enum_components = list(range(len(flat_event_shape)))
bijector = tfb.Restructure(
    enum_components,
    tf.nest.pack_sequence_as(target_model.event_shape, enum_components))(
        target_model.experimental_default_event_space_bijector())

current_state = bijector(
    tf.nest.map_structure(
        lambda e: tf.zeros([num_chains] + list(e), dtype=tf.float32),
    target_model.event_shape))

hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_model.unnormalized_log_prob,
    num_leapfrog_steps=num_leapfrog_steps,
    step_size=[tf.fill(s.shape, step_size) for s in current_state])

hmc = tfp.mcmc.TransformedTransitionKernel(
    hmc, bijector)

hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
    hmc,
    num_adaptation_steps=int(num_steps // 2 * 0.8),
    target_accept_prob=0.9)

chain, is_accepted = tf.function(
    lambda current_state: tfp.mcmc.sample_chain(
        current_state=current_state,
        kernel=hmc,
        num_results=num_steps // 2,
        num_burnin_steps=num_steps // 2,
        trace_fn=lambda _, pkr:
        (pkr.inner_results.inner_results.is_accepted),
        ),
    autograph=False,
    jit_compile=True)(current_state)

accept_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
ess = tf.nest.map_structure(
    lambda c: tfp.mcmc.effective_sample_size(
        c,
        cross_chain_dims=1,
        filter_beyond_positive_pairs=True),
    chain)

r_hat = tf.nest.map_structure(tfp.mcmc.potential_scale_reduction, chain)
hmc_samples = tf.nest.pack_sequence_as(target_model.event_shape, chain)
print('Acceptance rate is {}'.format(accept_rate))

chain_idx = 1 # one of the chains
ground_truth = []
idxs = []
i = 0
while 1:
  idx = random.randint(0, 10000)
  if is_accepted[idx][chain_idx] == True and idx not in idxs:
    ground_truth.append([
      hmc_samples[0][idx][chain_idx],
      hmc_samples[1][idx][chain_idx],
      hmc_samples[2][idx][chain_idx],
      hmc_samples[3][idx][chain_idx],
      hmc_samples[4][idx][chain_idx],
      hmc_samples[5][idx][chain_idx],
      hmc_samples[6][idx][chain_idx],
    ])
    idxs.append(idx)
    i+=1
  if i == 10:
    break

with open(f'ground_truth/radon/gt.pickle', 'wb') as handle:
  pickle.dump(ground_truth, handle, protocol=pickle.HIGHEST_PROTOCOL)