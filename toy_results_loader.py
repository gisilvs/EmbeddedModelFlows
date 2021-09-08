import pickle
import tensorflow as tf
import tensorflow_probability as tfp
from plot_utils import plot_heatmap_2d
import surrogate_posteriors

tfd = tfp.distributions
tfb = tfp.bijectors
Root = tfd.JointDistributionCoroutine.Root
data = 'checkerboard'
model = 'c20_sandwich'
with open(f'2d_toy_results/{data}/{model}.pickle', 'rb') as handle:
  results = pickle.load(handle)

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

