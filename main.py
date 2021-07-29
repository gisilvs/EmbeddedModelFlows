import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

import surrogate_posteriors
from models import get_model
from surrogate_posteriors import get_surrogate_posterior
from metrics import negative_elbo, forward_kl
from tensorflow_probability.python.internal import test_util

from plot_utils import plot_data
base_lr = 5e-5
target_lr = 1e-3
lr_scaling_factor = target_lr/base_lr

def scale_grad_by_factor(gradient_and_variable):
  gradient_and_variable = [(g[0]*lr_scaling_factor, g[1]) if 'residual_fraction' in g[1].name else (g[0], g[1]) for g in gradient_and_variable]
  return gradient_and_variable


model_name = 'lorenz_smoothing_r'
surrogate_posterior_name = 'gated_normalizing_program'
backbone_posterior_name= 'iaf'
seed = 80

prior, ground_truth, target_log_prob, observations = get_model(model_name, seed=seed)
surrogate_posterior = get_surrogate_posterior(prior, surrogate_posterior_name, backbone_posterior_name)

'''surrogate_posterior.log_prob(surrogate_posterior.sample())

with tf.GradientTape() as tape:
  posterior_sample = surrogate_posterior.sample(
    seed=(0, 0))
  posterior_logprob = surrogate_posterior.log_prob(posterior_sample)
grad = tape.gradient(posterior_logprob,
                     surrogate_posterior.trainable_variables)'''

# plot_data(model_name, ground_truth, observations)
trainable_variables = list(surrogate_posterior.trainable_variables)
print(surrogate_posteriors.residual_fraction_vars)

# todo: how do I save a fitted surrogate posterior (as if it was a neural network?)
losses = tfp.vi.fit_surrogate_posterior(target_log_prob,
                                        surrogate_posterior,
                                        optimizer=tf.optimizers.Adam(learning_rate=base_lr, gradient_transformers=[scale_grad_by_factor]),
                                        num_steps=100000,
                                        sample_size=50)

plt.plot(losses)
plt.show()
print(f'ELBO: {negative_elbo(target_log_prob, surrogate_posterior, num_samples=150, model_name=model_name, seed=seed)}')
print(f'FORWARD_KL: {forward_kl(surrogate_posterior, ground_truth)}')

print(surrogate_posteriors.residual_fraction_vars)