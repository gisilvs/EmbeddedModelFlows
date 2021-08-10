import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
import matplotlib.pyplot as plt

from surrogate_posteriors import get_surrogate_posterior

tfkl = tfk.layers
tfd = tfp.distributions
tfb = tfp.bijectors
tfe = tfp.experimental
Root = tfd.JointDistributionCoroutine.Root
is_bridge = False
is_classification = False

class AmortizerNet(tf.keras.Model):

  def __init__(self, output_shape,):
    super(AmortizerNet, self).__init__()
    # todo: find a way to extract only necessary variables
    self.dense1 = tfkl.Dense(units=50, activation=tf.nn.relu)
    self.dense2 = tfkl.Dense(units=output_shape , activation=None)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

@tfd.JointDistributionCoroutineAutoBatched
def model():
  innovation_noise= .1
  observation_noise = .15
  k = 5.
  truth = []
  new = yield Root(tfd.Normal(loc=0.,
                              scale=innovation_noise,
                              name='x_0'))
  truth.append(new)

  for t in range(1, 30):
    new = yield tfd.Normal(loc=new,
                           scale=innovation_noise,
                           name=f'x_{t}')
    truth.append(new)
  if is_bridge:
    time_steps = list(range(10)) + list(range(20,30))
  else:
    time_steps = range(30)
  for t in time_steps:
    if is_classification:
      yield tfd.Bernoulli(logits=k * truth[t], name=f'y_{t}')
    else:
      yield tfd.Normal(loc=truth[t],
                       scale=observation_noise,
                       name=f'y_{t}')


surrogate_posterior_name = 'mean_field'
backbone_posterior_name= 'iaf'
seed = 10

prior = model.experimental_pin(model.sample()[30:])
surrogate_posterior = get_surrogate_posterior(prior, surrogate_posterior_name, backbone_posterior_name)

def neg_elbo(surrogate_posterior, observations, posterior_samples):
  joint_dist = model.experimental_pin(observations)
  return - joint_dist.unnormalized_log_prob(posterior_samples) + surrogate_posterior.log_prob(posterior_samples)


num_epochs = 50
batch_size = 1

train_loss_results = []

flat_trainable_variables_len = len(tf.reshape(surrogate_posterior.trainable_variables, -1))

amortizer_net = AmortizerNet(flat_trainable_variables_len)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()

  observations = model.sample(batch_size)[30:]
  observations_tensor = tf.transpose(observations)
  amortized_trainable_variables = amortizer_net(observations_tensor)
  with tf.GradientTape() as tape:
    tape.watch(amortized_trainable_variables)
    # check if this works with other models
    for i, w in enumerate(surrogate_posterior.trainable_variables):
      w = amortized_trainable_variables[0, i]
    posterior_samples = surrogate_posterior.sample()  # how to deal with batch size here?
    loss_value = neg_elbo(surrogate_posterior, observations, posterior_samples)

  grads = tape.gradient(loss_value, amortizer_net.trainable_variables)
  optimizer.apply_gradients(zip(grads, amortizer_net.trainable_variables))
  epoch_loss_avg.update_state(loss_value)
  train_loss_results.append(epoch_loss_avg.result())
  if epoch % 5 == 0:
    print("Epoch {:03d}: Loss: {:.3f}".format(epoch,epoch_loss_avg.result()))

plt.plot(train_loss_results)
plt.show()

'''
def amortized_posterior(input_shape, flat_output_shape, event_shape_out):
  model = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape),
    ,
    ,
    tfb.Reshape(event_shape_out=event_shape_out),
  ])
  return model
'''
