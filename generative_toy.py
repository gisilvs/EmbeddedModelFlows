import tensorflow as tf
import tensorflow_probability as tfp

from toy_data import generate_2d_data
import surrogate_posteriors
from plot_utils import plot_heatmap_2d

import numpy as np
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root

n_components = 100
n_dims = 2
initial_scale = 1.
model_name = 'np_maf'

component_logits = tf.Variable([[1./n_components for _ in range(n_components)] for _ in
                      range(n_dims)])
locs = tf.Variable([tf.linspace(-4., 4., n_components) for _ in range(n_dims)])
scales = tfp.util.TransformedVariable([[initial_scale for _ in range(n_components)] for _ in
            range(n_dims)], tfb.Softplus())
'''component_logits = tf.convert_to_tensor([[1./n_components for _ in range(n_components)] for _ in
                      range(n_dims)])
locs = tf.convert_to_tensor([tf.linspace(-n_components/2, n_components/2, n_components) for _ in range(n_dims)])
scales = tf.convert_to_tensor([[.1 for _ in range(n_components)] for _ in
            range(n_dims)])'''

@tfd.JointDistributionCoroutine
def prior_structure():
  yield Root(tfd.Independent(tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(logits=component_logits),
      components_distribution=tfd.Normal(loc=locs, scale=scales),
      name=f"prior"), 1))

def grad(model, inputs, trainable_variables):
  with tf.GradientTape() as tape:
    loss = -model.log_prob(inputs)
  return loss, tape.gradient(loss, trainable_variables)



num_epochs = 1000
n = 1000
X, _ = generate_2d_data('8gaussians', batch_size=n)

'''plt.scatter(X[:,0], X[:,1], color='green')
plt.show()
'''
prior_matching_bijector = tfb.Chain(surrogate_posteriors._get_prior_matching_bijectors_and_event_dims(prior_structure)[-1])

maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, 'normalizing_program', 'maf')
# maf = surrogate_posteriors._sandwich_maf_normalizing_program(prior_structure)
maf.log_prob(prior_structure.sample())
trainable_variables=[]
trainable_variables.extend([component_logits, locs])
trainable_variables.extend(list(scales.trainable_variables))
trainable_variables.extend(list(maf.trainable_variables))

X_train = prior_matching_bijector(X)
dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(1000,reshuffle_each_iteration=True).padded_batch(64)

optimizer = tf.optimizers.Adam(learning_rate=1e-4)
train_loss_results = []

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  for x in dataset:
    # Optimize the model
    loss_value, grads = grad(maf, x, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))

    epoch_loss_avg.update_state(loss_value)

  train_loss_results.append(epoch_loss_avg.result())

  if epoch % 5 == 0:
    print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

plt.plot(train_loss_results)
plt.savefig(f'loss_c{n_components}_s{initial_scale}_{model_name}.png', format="png")
component_logits = tf.convert_to_tensor(component_logits)
locs = tf.convert_to_tensor(locs)
scales = tf.convert_to_tensor(scales)
plot_heatmap_2d(maf, matching_bijector=prior_matching_bijector, mesh_count=200, name=f'density_c{n_components}_s{initial_scale}_{model_name}.png')
'''x_ = tfkl.Input(shape=(2,), dtype=tf.float32)
log_prob_ = maf.log_prob(x_)
model = tfk.Model(x_, log_prob_)

model.compile(optimizer=tf.optimizers.Adam(), loss=lambda _, log_prob: -log_prob)
_ = model.fit(x=X,
              y=np.zeros((X.shape[0], 0), dtype=np.float32),
              batch_size=X.shape[0],
              epochs=2000,
              steps_per_epoch=1,
              verbose=1,
              shuffle=True)'''