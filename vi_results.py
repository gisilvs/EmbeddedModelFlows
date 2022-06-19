import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from models.model_getter import get_surrogate_posterior, _get_prior_matching_bijectors_and_event_dims
from vi.models import get_model
import seaborn as sns

rc = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"]}
plt.rcParams.update(rc)


def kernel_density(samples, t=14, name='kde.png', legend=True):
  samples = samples[t]
  sns.distplot(samples[:, 0], hist=False, kde=True,
               kde_kws={'shade': True}, label='x')
  sns.distplot(samples[:, 1], hist=False, kde=True,
               kde_kws={'shade': True}, label='y')
  p = sns.distplot(samples[:, 2], hist=False, kde=True,
               kde_kws={'shade': True}, label='z')
  if not legend:
    p.set(ylabel=None)
  sns.despine()
  if legend:
    plt.legend()
  plt.savefig(name, transparent=True)
  plt.close()

tfb = tfp.bijectors

i = 0
seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
model_name = 'lorenz_smoothing_r'
surrogate_posterior_name = 'gated_normalizing_program'
backbone_name = 'iaf'
prior, ground_truth, target_log_prob, observations = get_model(
      model_name, seed=seeds[i])
surrogate_posterior = get_surrogate_posterior(prior, surrogate_posterior_name, backbone_name)
checkpoint = tf.train.Checkpoint(weights=surrogate_posterior.trainable_variables)
repo_name = f'results_0/{model_name}/{surrogate_posterior_name}'
ckpt_dir = f'{repo_name}/checkpoints_{i}'
checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
prior_matching_bijector = tfb.Chain(_get_prior_matching_bijectors_and_event_dims(prior)[-1])

x = surrogate_posterior.distribution.distribution.sample(10000)
x0 = tf.convert_to_tensor(prior_matching_bijector(x))
kernel_density(x0, name='kde_noise.png')
plt.plot(x0[:,0,:])
plt.axis('off')
plt.savefig('noise.png', transparent=True)
plt.close()
y = surrogate_posterior.distribution.bijector.forward(x)
y0 = tf.convert_to_tensor(y)
kernel_density(y0, name='kde_iaf.png', legend=False)
plt.plot(y0[:,0,:])
plt.axis('off')
plt.savefig('iaf.png', transparent=True)
plt.close()
z = surrogate_posterior.bijector.forward(y)
z0 = tf.convert_to_tensor(z)
kernel_density(z0, name='kde_mef.png', legend=False)
plt.plot(z0[:,0,:])
plt.axis('off')
plt.savefig('mef.png', transparent=True)
plt.close()
