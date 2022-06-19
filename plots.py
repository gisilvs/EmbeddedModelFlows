import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root
n_components = 5
n_dims = 1
component_logits = tf.convert_to_tensor(
          [[1. / n_components for _ in range(n_components)] for _ in
           range(n_dims)])
locs = tf.convert_to_tensor(
          [tf.linspace(-3., 3., n_components) for _
           in
           range(n_dims)])
scales = tf.convert_to_tensor(
          [[.2 for _ in range(n_components)] for _ in
           range(n_dims)])

'''@tfd.JointDistributionCoroutine
def prior_structure():
  yield Root(tfd.Independent(tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(logits=component_logits),
    components_distribution=tfd.Normal(loc=locs, scale=scales),
    name=f"prior"), 1))'''

@tfd.JointDistributionCoroutine
def prior_structure():
  z = yield Root(tfd.Normal(0.,3.))
  x = yield tfd.Normal(0., tf.math.exp(z/2))


emf = surrogate_posteriors.GatedAutoFromNormal(prior_structure)
prior_matching_bijector = \
  surrogate_posteriors._get_prior_matching_bijectors_and_event_dims(
    prior_structure)[-1]
base_distribution = tfd.Sample(
  tfd.Normal(tf.zeros([], dtype=tf.float32), 1.), sample_shape=[2])

new_dist = tfd.TransformedDistribution(
    distribution=base_distribution,
    bijector=tfb.Chain([emf] + prior_matching_bijector)
  )

base_samples = base_distribution.sample(10000)
'''plt.scatter(base_samples[:,0], base_samples[:,1], s=5, vmin=-4, vmax=4)
plt.xlim(-4,  4)
plt.ylim(-4, 4)
plt.show()'''
transformed_samples = tf.squeeze(tf.convert_to_tensor(new_dist.bijector.forward(
  base_samples)))
plt.scatter(transformed_samples[1,:], transformed_samples[0, :], s=5)
plt.title('$\lambda=0.001$')
plt.xlim(-4,  4)
plt.ylim(-4, 4)
plt.show()
a = 0