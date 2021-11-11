import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root

@tfd.JointDistributionCoroutine
def lorenz_system():
  truth = []
  innovation_noise = .1
  step_size = 0.02
  loc = yield Root(tfd.Sample(tfd.Normal(0., 1., name='x_0'), sample_shape=3))
  for t in range(1, 30):
    x, y, z = tf.unstack(loc, axis=-1)
    truth.append(x)
    dx = 10 * (y - x)
    dy = x * (28 - z) - y
    dz = x * y - 8 / 3 * z
    delta = tf.stack([dx, dy, dz], axis=-1)
    loc = yield tfd.Independent(
      tfd.Normal(loc + step_size * delta,
                 tf.sqrt(step_size) * innovation_noise, name=f'x_{t}'),
      reinterpreted_batch_ndims=1)

@tfd.JointDistributionCoroutine
def brownian_motion():
  new = yield Root(tfd.Normal(loc=0, scale=.1))

  for t in range(1, 30):
    new = yield tfd.Normal(loc=new, scale=.1)

@tfd.JointDistributionCoroutine
def ornstein_uhlenbeck():
  a = 0.8
  new = yield Root(tfd.Normal(loc=0, scale=5.))

  for t in range(1, 30):
    new = yield tfd.Normal(loc=a*new, scale=.5)

rc = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"]}
plt.rcParams.update(rc)
#plt.rcParams["figure.figsize"] = (9,3)

models = ['np_maf_continuity', 'np_maf_smoothness', 'maf','maf3', 'bottom']
data = ['brownian', 'ornstein', 'lorenz']
model = 'np_maf_stock'
run = 'run_0'
d_names = {
  'brownian': 'Brownian motion',
  'ornstein': 'Ornstein-Uhlenbeck process',
  'lorenz': 'Lorenz system'
}

'''d_names = {
  brownian_motion: 'Brownian motion',
  ornstein_uhlenbeck: 'Ornstein-Uhlenbeck process',
  lorenz_system: 'Lorenz system'
}'''

names = {
  'maf': 'MAF',
  'bottom': 'B-MAF',
  'np_maf_continuity': 'GEMF-T(c)',
  'np_maf_smoothness': 'GEMF-T(s)',
  'maf3': 'MAF-L'
}

'''generators=[lorenz_system]
for g in generators:
  samples = tf.convert_to_tensor(g.sample(10))
  plt.plot(samples[:, :10, 0], alpha=0.7, linewidth=1)
  plt.title(f'{d_names[g]} - Ground truth')
  plt.savefig(f'time_series_results/samples_{d_names[g]}_gt.png')
  plt.close()'''

for d in data:
  for model in models:
    with open(f'time_series_results/{run}/{d}/{model}.pickle', 'rb') as handle:
      results = pickle.load(handle)
    plt.plot(results['loss'], label=names[model])
  if d == 'lorenz':
    plt.ylim(bottom=-300, top=800)
  plt.title(d_names[d])
  plt.legend()
  plt.savefig(f'time_series_results/loss_{d}.png')
  plt.close()
  '''samples = results['samples']
  plt.plot(samples[:, :10, 0], alpha=0.7, linewidth=1)
  plt.title(f'{d_names[d]} - {names[model]}')
  plt.savefig(f'time_series_results/samples_{d}_{model}.png')
  plt.close()'''

'''run = 0
if model == 'ground_truth':
  with open(f'time_series_results/ground_truth/{data}.pickle',
            'rb') as handle:
    results = pickle.load(handle)
else:
  with open(f'time_series_results/run_{run}/{data}/{model}.pickle', 'rb') as handle:
    results = pickle.load(handle)


samples = results['samples']
samples = tf.reshape(tf.transpose(tf.convert_to_tensor(samples)),[-1,40,2])
plt.plot(samples[:,:10,0], alpha=0.7, linewidth=1)
plt.show()'''