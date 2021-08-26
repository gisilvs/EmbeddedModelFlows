import tensorflow_datasets as tfds
import tensorflow as tf
import pixelcnn_original

tfk = tf.keras
tfkl = tf.keras.layers

image_side_size = 8
# Load MNIST from tensorflow_datasets
data = tfds.load("mnist", split=["train", "test"])
train_data, test_data = data[0], data[1]


def image_preprocess(x):
  x['image'] = tf.cast(x['image'], tf.float32)
  x['image'] = tf.image.resize(x['image'], [image_side_size, image_side_size])
  return (x['image'],)  # (input, output) of the model


batch_size = 16
train_it = train_data.map(image_preprocess).batch(batch_size).shuffle(1000)

image_shape = (image_side_size, image_side_size, 1)
# Define a Pixel CNN network
dist = pixelcnn_original.PixelCNN(
  image_shape=image_shape,
  num_resnet=1,
  num_hierarchies=2,
  num_filters=32,
  num_logistic_mix=5,
  dropout_p=.3,
  use_weight_norm=False,
)

# Define the model input
image_input = tfkl.Input(shape=image_shape)

# Define the log likelihood for the loss fn
log_prob = dist.log_prob(image_input)

# Define the model
model = tfk.Model(inputs=image_input, outputs=log_prob)
model.add_loss(-tf.reduce_mean(log_prob))

# Compile and train the model
model.compile(
  optimizer=tfk.optimizers.Adam(.001),
  metrics=[])

model.fit(train_it, epochs=20, verbose=True)

dist.network.save_weights(f'MNIST_{image_side_size}', save_format='tf')