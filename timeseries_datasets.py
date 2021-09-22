from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_mauna_loa_atmospheric_co2():
  ml_data = fetch_openml(data_id=41187, as_frame=False)
  months = []
  ppmv_sums = []
  counts = []

  y = ml_data.data[:, 0]
  m = ml_data.data[:, 1]
  month_float = y + (m - 1) / 12
  ppmvs = ml_data.target

  for month, ppmv in zip(month_float, ppmvs):
    if not months or month != months[-1]:
      months.append(month)
      ppmv_sums.append(ppmv)
      counts.append(1)
    else:
      # aggregate monthly sum to produce average
      ppmv_sums[-1] += ppmv
      counts[-1] += 1

  months = np.asarray(months).reshape(-1, 1)
  avg_ppmvs = np.asarray(ppmv_sums) / counts

  avg_ppmvs_by_year = np.array([avg_ppmvs[i-24:i] for i in list(range(24,522))])
  means = np.mean(avg_ppmvs_by_year, axis=1).reshape(-1,1)
  stds = np.std(avg_ppmvs_by_year, axis=1).reshape(-1,1)
  normalized_avg_ppmvs_by_year = (avg_ppmvs_by_year-means)/stds
  train = tf.convert_to_tensor(normalized_avg_ppmvs_by_year[:400])
  valid = tf.convert_to_tensor(normalized_avg_ppmvs_by_year[400:448])
  test = tf.convert_to_tensor(normalized_avg_ppmvs_by_year[448:])
  return tf.cast(train, tf.float32), tf.cast(valid, tf.float32), tf.cast(test, tf.float32)