import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from models import get_model
from surrogate_posteriors import get_surrogate_posterior
from metrics import negative_elbo, forward_kl
from tensorflow_probability.python.internal import test_util

from plot_utils import plot_data

model_name = 'lorenz_smoothing_r'
surrogate_posterior_name = 'gated_normalizing_program'
backbone_posterior_name= 'mean_field'

prior, ground_truth, target_log_prob, observations = get_model(model_name, seed=10)
surrogate_posterior = get_surrogate_posterior(prior, surrogate_posterior_name, backbone_posterior_name)

surrogate_posterior.log_prob(surrogate_posterior.sample())

with tf.GradientTape() as tape:
  posterior_sample = surrogate_posterior.sample(
    seed=(0, 0))
  posterior_logprob = surrogate_posterior.log_prob(posterior_sample)
grad = tape.gradient(posterior_logprob,
                     surrogate_posterior.trainable_variables)

# plot_data(model_name, ground_truth, observations)
# todo: how do I save a fitted surrogate posterior (as if it was a neural network?)
losses = tfp.vi.fit_surrogate_posterior(target_log_prob,
                                        surrogate_posterior,
                                        optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                                        num_steps=20000,
                                        sample_size=50,
                                        trainable_variables=surrogate_posterior.trainable_variables)

plt.plot(losses)
plt.show()
print(f'ELBO: {negative_elbo(target_log_prob, surrogate_posterior, num_samples=150, model_name=model_name, seed=10)}')
print(f'FORWARD_KL: {forward_kl(surrogate_posterior, ground_truth)}')

