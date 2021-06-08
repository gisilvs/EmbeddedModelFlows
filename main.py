import tensorflow as tf
import tensorflow_probability as tfp

from models import get_model
from surrogate_posteriors import get_surrogate_posterior
from metrics import negative_elbo, forward_kl

model_name = 'eight_schools'
surrogate_posterior_name = 'normalizing_program'
backbone_posterior_name= 'highway_flow'

model, prior, ground_truth, target_log_prob, observations = get_model(model_name)
surrogate_posterior = get_surrogate_posterior(prior, surrogate_posterior_name, backbone_posterior_name)
# todo: how do I save a fitted surrogate posterior (as if it was a neural network?)
losses = tfp.vi.fit_surrogate_posterior(target_log_prob,
                                        surrogate_posterior,
                                        optimizer=tf.optimizers.Adam(learning_rate=1e-2),
                                        num_steps=50,
                                        sample_size=10)

print(f'ELBO: {negative_elbo(target_log_prob, surrogate_posterior, num_smaples=15, model_name=model_name)}')
print(f'FORWARD_KL: {forward_kl(surrogate_posterior, ground_truth)}')