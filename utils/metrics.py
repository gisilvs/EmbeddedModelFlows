import tensorflow as tf


def negative_elbo(target_log_prob, surrogate_posterior, num_samples, seed):
  samples = surrogate_posterior.sample(num_samples, seed=seed)
  return - tf.reduce_mean(
    target_log_prob(*samples) - surrogate_posterior.log_prob(samples))


def forward_kl(surrogate_posterior, ground_truth):
  return surrogate_posterior.log_prob(ground_truth)
