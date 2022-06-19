import pickle
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from vi.models import get_vi_model

num_results = 5000
num_burnin_steps = 3000

num_schools = 8

_, _, target_log_prob_fn, _ = get_vi_model('eight_schools', seed=10)


# Improve performance by tracing the sampler using `tf.function`
# and compiling it using XLA.
@tf.function(autograph=False, experimental_compile=True)
def do_sampling():
  return tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=[
      tf.zeros([], name='init_avg_effect'),
      tf.zeros([], name='init_avg_stddev'),
      tf.ones([num_schools], name='init_school_effects_standard'),
    ],
    kernel=tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target_log_prob_fn,
      step_size=0.4,
      num_leapfrog_steps=3))


states, kernel_results = do_sampling()

avg_effect, avg_stddev, school_effects_standard = states

school_effects_samples = (
    avg_effect[:, np.newaxis] +
    np.exp(avg_stddev)[:, np.newaxis] * school_effects_standard)

num_accepted = np.sum(kernel_results.is_accepted)
print('Acceptance rate: {}'.format(num_accepted / num_results))

ground_truth = []
idxs = []
i = 0
while 1:
  idx = random.randint(0, 5000)
  if kernel_results.is_accepted[idx] == True and idx not in idxs:
    ground_truth.append([
      avg_effect[idx],
      avg_stddev[idx],
      school_effects_standard[idx],
    ])
    idxs.append(idx)
    i += 1
  if i == 10:
    break

with open(f'../ground_truth/eight_schools/gt.pickle', 'wb') as handle:
  pickle.dump(ground_truth, handle, protocol=pickle.HIGHEST_PROTOCOL)
