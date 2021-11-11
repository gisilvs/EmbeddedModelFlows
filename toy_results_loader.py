import pickle
import tensorflow as tf
import tensorflow_probability as tfp
from plot_utils import plot_heatmap_2d
import surrogate_posteriors
import matplotlib.pyplot as plt

rc = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"]}
plt.rcParams.update(rc)

tfd = tfp.distributions
tfb = tfp.bijectors
Root = tfd.JointDistributionCoroutine.Root
run = 'run_0'
data = ['brownian', 'ornstein', 'lorenz']
models = ['maf', 'bottom', 'np_maf_continuity', 'np_maf_smoothness']
d_names = {
  'brownian': 'Brownian motion',
  'ornstein': 'Ornstein-Uhlenbeck process',
  'lorenz': 'Lorenz system'
}
names = {
  'maf': 'MAF',
  'bottom': 'MEF-B',
  'np_maf_continuity': 'GMEF-T(c)',
  'np_maf_smoothness': 'GMEF-T(s)'
}
for d in data:
  if d == 'lorenz':
    plt.ylim(bottom=-800, top=800)
  for model in models:
    with open(f'time_series_results/{run}/{d}/{model}.pickle', 'rb') as handle:
      results = pickle.load(handle)
    plt.plot(results['loss'], label=names[model])

  plt.title(d)
  plt.legend(loc='upper right')
  plt.savefig(f'time_series_results/loss_{d}.png')
  plt.close()

'''a = 0
@tfd.JointDistributionCoroutine
def prior_structure():
  yield Root(tfd.Independent(tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(logits=results['component_logits']),
    components_distribution=tfd.Normal(loc=results['locs'], scale=results['scales']),
    name=f"prior"), 1))

prior_matching_bijector = tfb.Chain(
    surrogate_posteriors._get_prior_matching_bijectors_and_event_dims(
      prior_structure)[-1])

plot_heatmap_2d(prior_structure, prior_matching_bijector, xmin=-6., xmax=6., ymin=-6., ymax=6.)
'''
