import pickle

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root

n_dims = 784
lambd = 1e-6
to_plot=True
to_save=False


names = {
  'maf': 'MAF',
  'maf_bn': 'MAF-an',
  'maf3': 'MAF-L',
  'np_maf': 'EMF-T',
  'np_maf_bn': 'EMF-T-an',
  'c100_np_maf': 'EMF-T',
  'sandwich': 'EMF-M',
  'c100_sandwich': 'EMF-M',
  'np_maf_smoothness': 'EMF-T(s)',
  'np_maf_continuity': 'EMF-T(c)',
  'bottom': 'B-MAF',
  'splines': 'NSF',
  'c100_np_splines': 'NSF-EMF-T',
  'c100_sandwich_splines': 'NSF-EMF-M',
  'np_splines': 'NSF-EMF-T',
}


def inverse_logits(x):
  x = tfb.Sigmoid()(x)
  x = x / (1 - 2 * lambd) - lambd
  return x

@tf.function
def eval(model, inputs):
  return -model.log_prob(inputs)

@tf.function
def sample(model, model_fixed, n_samples, model_name):
  if model_name in ['np_maf', 'np_maf_bn', 'np_splines']:
    x = model.distribution.sample(int(n_samples))
    '''for i in reversed(range(len(model.distribution.bijector.bijectors))):
      x = model.distribution.bijector.bijectors[i].forward(x)'''
    x = model_fixed.bijector.forward(x)
    return x

  elif model_name in ['sandwich', 'sandwich_bn']:
    x = model.distribution.sample(int(n_samples))
    for i in reversed(range(len(model_fixed.bijector.bijectors))):
      bij_name = model_fixed.bijector.bijectors[i].name
      if 'chain' in bij_name:
        x = model_fixed.bijector.bijectors[i].forward(x)
      else:
        x = model.bijector.bijectors[i].forward(x)
    return x

@tf.function
def get_forward(x, model, model_fixed, model_name):
  if model_name in ['np_maf', 'np_maf_bn']:
    for i in reversed(range(len(model.distribution.bijector.bijectors))):
      x = model.distribution.bijector.bijectors[i].forward(x)
    x = model_fixed.bijector.forward(x)
    return x

  elif model_name in ['maf', 'splines']:
    for i in reversed(range(len(model.bijector.bijectors))):
      x = model.bijector.bijectors[i].forward(x)
    return x

@tf.function
def get_inverse(x, model, model_name):
  if model_name in ['np_maf', 'np_maf_bn']:
    x = model.bijector.inverse(x)
    for i in range(len(model.distribution.bijector.bijectors)):
      x = model.distribution.bijector.bijectors[i].inverse(x)
    return x

  elif model_name in ['maf', 'splines']:
    for i in range(len(model.bijector.bijectors)):
      x = model.bijector.bijectors[i].inverse(x)
    return x

  '''elif model_name in ['sandwich', 'sandwich_bn']:
    x = model.distribution.sample(int(n_samples))
    for i in reversed(range(len(model_fixed.bijector.bijectors))):
      bij_name = model_fixed.bijector.bijectors[i].name
      if 'chain' in bij_name:
        x = model_fixed.bijector.bijectors[i].forward(x)
      else:
        x = model.bijector.bijectors[i].forward(x)
    return x'''

def get_samples(model, n_components, name, save_dir):
  def build_model(model_name, trainable_mixture=True, component_logits=None,
                  locs=None, scales=None):
    if trainable_mixture:
      if model_name in ['maf', 'splines', 'maf3', 'maf_bn', 'splines_bn',
                        'maf3_bn']:
        component_logits = tf.convert_to_tensor(
          [[1. / n_components for _ in range(n_components)] for _ in
           range(n_dims)])
        locs = tf.convert_to_tensor(
          [tf.linspace(-n_components / 2, n_components / 2, n_components) for _
           in
           range(n_dims)])
        scales = tf.convert_to_tensor(
          [[1. for _ in range(n_components)] for _ in
           range(n_dims)])
      else:
        if model_name in ['np_maf', 'np_splines', 'np_maf_bn', 'np_splines_bn']:
          loc_range = 15.
          scale = 3.
        else:
          loc_range = 20.
          scale = 1.
        component_logits = tf.Variable(
          [[1. / n_components for _ in range(n_components)] for _ in
           range(n_dims)], name='component_logits')
        if 'sandwich' in model_name:
          locs = tf.Variable(
            [tf.linspace(-loc_range, loc_range, n_components) for _ in
             range(n_dims)],
            name='locs')
        else:
          locs = tf.Variable(
            [tf.linspace(-loc_range, 0., n_components) for _ in
             range(n_dims)],
            name='locs')
        scales = tfp.util.TransformedVariable(
          [[scale for _ in range(n_components)] for _ in
           range(n_dims)], tfb.Softplus(), name='scales')

    @tfd.JointDistributionCoroutine
    def prior_structure():
      yield Root(tfd.Independent(tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=component_logits),
        components_distribution=tfd.Normal(loc=locs, scale=scales),
        name=f"prior"), 1))

    prior_matching_bijector = tfb.Chain(
      surrogate_posteriors._get_prior_matching_bijectors_and_event_dims(
        prior_structure)[-1])
    use_bn = False
    if '_bn' in model_name:
      use_bn = True
    if model_name in ['maf', 'maf_bn']:
      flow_params = {'use_bn': use_bn}
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure,
                                                         'maf',
                                                         flow_params=flow_params)

    elif model_name in ['maf3', 'maf3_bn']:
      flow_params = {'num_flow_layers': 3,
                     'use_bn': use_bn}
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure, 'maf',
                                                         flow_params=flow_params)
    elif model_name in ['np_maf', 'np_maf_bn']:
      flow_params = {'use_bn': use_bn}
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure,
                                                         'normalizing_program',
                                                         'maf',
                                                         flow_params=flow_params)
    elif model_name in ['sandwich', 'sandwich_bn']:
      maf = surrogate_posteriors._sandwich_maf_normalizing_program(
        prior_structure, use_bn=use_bn)

    elif model_name in ['sandwich_splines', 'sandwich_splines_bn']:
      flow_params = {
        'layers': 6,
        'number_of_bins': 32,
        'input_dim': 784,
        'nn_layers': [32, 32],
        'b_interval': 15,
        'use_bn': use_bn
      }
      maf = surrogate_posteriors._sandwich_splines_normalizing_program(
        prior_structure, flow_params=flow_params)

    elif model_name in ['splines', 'splines_bn']:
      flow_params = {
        'layers': 6,
        'number_of_bins': 32,
        'input_dim': 784,
        'nn_layers': [32, 32],
        'b_interval': 15,
        'use_bn': use_bn
      }
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure,
                                                         surrogate_posterior_name='splines',
                                                         flow_params=flow_params)
    elif model_name in ['np_splines', 'np_splines_bn']:
      flow_params = {
        'layers': 6,
        'number_of_bins': 32,
        'input_dim': 784,
        'nn_layers': [32, 32],
        'b_interval': 15,
        'use_bn': use_bn
      }
      maf = surrogate_posteriors.get_surrogate_posterior(prior_structure,
                                                         surrogate_posterior_name='normalizing_program',
                                                         backnone_name='splines',
                                                         flow_params=flow_params)

    return maf, prior_matching_bijector

  def _preprocess(sample):
    image = tf.cast(sample['image'], tf.float32)
    image = (image + tf.random.uniform(tf.shape(image), minval=0., maxval=1.,
                                       seed=42)) / 256.  # dequantize
    image = lambd + (1 - 2 * lambd) * image
    image = tfb.Invert(tfb.Sigmoid())(image)  # logit
    image = tf.reshape(image, [-1])
    image = prior_matching_bijector(image)
    return image

  flow, prior_matching_bijector = build_model(model)

  data = tfds.load("mnist", split=["train[:50000]", "train[50000:]", "test"])
  train_data, valid_data, test_data = data[0], data[1], data[2]

  test_dataset = (test_data
                  .map(map_func=_preprocess,
                       num_parallel_calls=tf.data.AUTOTUNE)
                  .cache()
                  .batch(256)
                  .prefetch(tf.data.AUTOTUNE))
  x = next(iter(test_dataset))

  ckpt_dir = f'{main_dir}/run_{run}/checkpoints/{name}'
  flow.log_prob(x)
  eval(flow, x)

  checkpoint = tf.train.Checkpoint(weights=flow.trainable_variables)
  checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
  # samples = flow.sample(5)

  '''for i in range(len(flow.bijector.bijectors)):
    if flow.bijector.bijectors[i].name == 'batch_normalization':
      flow.bijector.bijectors[i].batchnorm.trainable=False
      flow.bijector.bijectors[i].batchnorm.moving_mean = tf.zeros_like(
        flow.bijector.bijectors[i].batchnorm.moving_mean)
      flow.bijector.bijectors[i].batchnorm.moving_variance = tf.ones_like(
        flow.bijector.bijectors[i].batchnorm.moving_variance)'''

  if model in ['sandwich', 'sandwich_bn', 'np_maf', 'np_maf_bn', 'np_splines']:
      for v in flow.trainable_variables:
        if 'locs' in v.name:
          locs = tf.convert_to_tensor(v)
        elif 'scales' in v.name:
          scales = tf.math.softplus(tf.convert_to_tensor(v))
        elif 'component_logits' in v.name:
          component_logits = tf.convert_to_tensor(v)

      fixed_flow, _ = build_model(model, trainable_mixture=False,
                                 component_logits=component_logits, locs=locs,
                                 scales=scales)

      samples = sample(flow, fixed_flow, 16, model)
  else:
    samples = flow.sample(16)

  '''data = tfds.load("mnist",
                   split=["train[:50000]", "train[50000:]", "test"])
  train_data, valid_data, test_data = data[0], data[1], data[2]

  train_dataset = (train_data
                   .map(_preprocess)
                   .cache()
                   # .shuffle(int(10e3))
                   .batch(9)
                   .prefetch(tf.data.AUTOTUNE))

  x = next(iter(train_dataset))
  inv = get_inverse(x, flow, model)
  back = get_forward(tf.identity(inv), flow, fixed_flow, model)'''

  a = 0

  if to_save:
    with open(f'{save_dir}/samples_{name}.pickle', 'wb') as handle:
      pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)


  if to_plot:
    samples = inverse_logits(samples)
    samples = tf.reshape(samples, [-1, 28, 28])
    _, axs = plt.subplots(4, 4,)
    axs = axs.flatten()
    for img, ax in zip(samples, axs):
      ax.axis('off')
      ax.imshow(img, cmap='gray', vmin=0., vmax=1.)
    plt.suptitle(f'{names[model]}')
    plt.savefig(f'mnist/samples_{model}.png', transparent=True)
    plt.close()


  print(f'{name} done!')

models = ['sandwich']#, 'np_maf', 'sandwich',
# 'maf',
# 'maf3']

main_dir = 'mnist'
n_runs = [0]

for run in n_runs:
  for model in models:
    if model == 'maf' or model == 'maf_bn':
      get_samples(model, 20, model, save_dir=f'{main_dir}/run_{run}')
    elif model == 'maf3':
      name = 'maf3'
      get_samples(model, 20, name, save_dir=f'{main_dir}/run_{run}')
    elif model == 'splines' or model == 'splines_bn':
      get_samples(model, 20, model, save_dir=f'{main_dir}/run_{run}')
    else:
      for n_components in [100]:
        name = f'c{n_components}_{model}'
        get_samples(model, n_components, name, save_dir=f'{main_dir}/run_{run}')