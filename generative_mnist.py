import os
import shutil
import pickle
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers
Root = tfd.JointDistributionCoroutine.Root

n_components = 100
lambd = 1e-6
n_dims = 784
num_epochs = 10000000

def clear_folder(folder):
  for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
      elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
    except Exception as e:
      print('Failed to delete %s. Reason: %s' % (file_path, e))

def train(model, n_components, name, save_dir):
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
            [tf.linspace(-loc_range, loc_range, n_components) for _ in
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
                                                         'maf', flow_params=flow_params)

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
        'layers': 3,
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
        'nn_layers': [32,32],
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

    # maf.log_prob(prior_structure.sample(2))

    return maf, prior_matching_bijector


  @tf.function
  def optimizer_step(net, inputs):
    with tf.GradientTape() as tape:
      loss = -net.log_prob(inputs)
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss

  @tf.function
  def eval(model, inputs):
    return -model.log_prob(inputs)

  def inverse_logits(x):
    x = tfb.Sigmoid()(x)
    x = x / (1 - 2 * lambd) - lambd
    return x

  def _preprocess(sample):
    image = tf.cast(sample['image'], tf.float32)
    image = (image + tf.random.uniform(tf.shape(image), minval=0., maxval=1.,
                                       seed=42)) / 256.  # dequantize
    image = lambd + (1 - 2 * lambd) * image
    image = tfb.Invert(tfb.Sigmoid())(image) # logit
    image = tf.reshape(image, [-1])
    image = prior_matching_bijector(image)
    return image

  maf, prior_matching_bijector = build_model(model)
  data = tfds.load("mnist", split=["train[:50000]", "train[50000:]", "test"])
  train_data, valid_data, test_data = data[0], data[1], data[2]

  train_dataset = (train_data
                   .map(map_func=_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                   .cache()
                   .shuffle(int(1e3))
                   .batch(256)
                   .prefetch(tf.data.AUTOTUNE))

  valid_dataset = (valid_data
                   .map(map_func=_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                   .cache()
                   .batch(256)
                   .prefetch(tf.data.AUTOTUNE))

  test_dataset = (test_data
                   .map(map_func=_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                   .cache()
                   .batch(256)
                   .prefetch(tf.data.AUTOTUNE))


  optimizer = tf.optimizers.Adam(learning_rate=1e-4)
  train_loss_results = []
  valid_loss_results = []
  best_val_loss = None
  epochs_counter = 0

  for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    valid_loss_avg = tf.keras.metrics.Mean()
    for x in train_dataset:
      # Optimize the model
      loss_value = optimizer_step(maf, x)
      #print(loss_value)
      epoch_loss_avg.update_state(loss_value)
    train_loss_results.append(epoch_loss_avg.result())
    if epoch % 100 == 0:
      print(epoch)
      print(train_loss_results[-1])
    for x in valid_dataset:
      loss_value = eval(maf, x)
      valid_loss_avg.update_state(loss_value)
    valid_loss_results.append(valid_loss_avg.result())

    if tf.math.is_nan(valid_loss_avg.result()):
      break

    if epoch == 0:
      checkpoint = tf.train.Checkpoint(weights=maf.trainable_variables)
      ckpt_dir = f'/tmp/{save_dir}/checkpoints/{name}'
      if os.path.isdir(ckpt_dir):
        clear_folder(ckpt_dir)
      checkpoint_manager = tf.train.CheckpointManager(checkpoint, ckpt_dir,
                                                      max_to_keep=20)
      best_loss = valid_loss_avg.result()

    elif best_loss > valid_loss_avg.result():
      test_loss_avg = tf.keras.metrics.Mean()
      for x in test_dataset:
        loss_value = eval(maf, x)
        test_loss_avg.update_state(loss_value)
      if not tf.math.is_nan(test_loss_avg.result()):
        save_path = checkpoint_manager.save()
      best_loss = valid_loss_avg.result()
      epochs_counter = 0
    else:
      epochs_counter +=1
      if epochs_counter > 100:
        print(f'Stop at epoch {epoch}')
        break

  new_maf, _ = build_model(model)
  new_maf.log_prob(x)
  eval(new_maf, x)
  new_checkpoint = tf.train.Checkpoint(weights=new_maf.trainable_variables)

  new_checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  if os.path.isdir(f'{save_dir}/checkpoints/{name}'):
    clear_folder(f'{save_dir}/checkpoints/{name}')
  checkpoint_manager = tf.train.CheckpointManager(new_checkpoint,
                                                  f'{save_dir}/checkpoints/{name}',
                                                  max_to_keep=20)
  save_path = checkpoint_manager.save()

  plt.plot(train_loss_results)
  plt.plot(valid_loss_results)
  plt.savefig(f'{save_dir}/loss_{name}.png',
              format="png")
  plt.close()

  test_loss_avg = tf.keras.metrics.Mean()

  for x in test_dataset:
    loss_value = eval(new_maf, x)
    test_loss_avg.update_state(loss_value)

  results = {#'samples': tf.convert_to_tensor(new_maf.sample(1000)),
             'loss_eval': test_loss_avg.result(),
             'train_loss': train_loss_results,
             'valid_loss': valid_loss_results
             }

  with open(f'{save_dir}/{name}.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print(f'{name} done!')

models = [
  #'np_maf',
  #'maf',
  #'maf3'
  #'splines'
  #'np_splines'
  'sandwich_bn',
  #'sandwich_splines_bn',
  ]
# 'np_maf',
# 'sandwich',
# 'maf',
# 'maf3']

main_dir = 'mnist'
if not os.path.isdir(main_dir):
  os.makedirs(main_dir)
n_runs = [3]

for run in n_runs:
  if not os.path.exists(f'{main_dir}/run_{run}'):
    os.makedirs(f'{main_dir}/run_{run}')
  for model in models:
    if model == 'maf' or model == 'maf_bn':
      train(model, 20, model, save_dir=f'{main_dir}/run_{run}')
    elif model == 'maf3':
      name = 'maf3'
      train(model, 20, name, save_dir=f'{main_dir}/run_{run}')
    elif model == 'splines' or model == 'splines_bn':
      train(model, 20, model, save_dir=f'{main_dir}/run_{run}')
    else:
      for n_components in [100]:
        name = f'c{n_components}_{model}'
        train(model, n_components, name, save_dir=f'{main_dir}/run_{run}')