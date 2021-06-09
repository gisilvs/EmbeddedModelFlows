import tensorflow as tf

def negative_elbo(target_log_prob, surrogate_posterior, num_smaples, model_name):
  samples = surrogate_posterior.sample(num_smaples, seed=45)
  if model_name == 'eight_schools':
    return - tf.reduce_mean(target_log_prob(**samples) - surrogate_posterior.log_prob(samples))
  else:
    return - tf.reduce_mean(
      target_log_prob(*samples) - surrogate_posterior.log_prob(samples))

def forward_kl(surrogate_posterior, ground_truth):
  return surrogate_posterior.log_prob(ground_truth)

