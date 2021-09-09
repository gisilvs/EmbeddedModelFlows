import tensorflow_datasets as tfds
import tensorflow as tf

def _preprocess(sample):

  image = tf.cast(sample['image'], tf.float32)
  image = image + tf.random.uniform(tf.shape(image), minval=0, maxval=256)
  return image

data = tfds.load("mnist", split=["train[:90%]", "train[90%:]", "test"])
train_data, valid_data, test_data = data[0], data[1], data[2]

train_dataset = (train_data
                 .map(_preprocess)
                 .batch(256)
                 .prefetch(tf.data.AUTOTUNE)
                 .shuffle(int(10e3)))

a = 0