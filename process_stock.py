import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_stock_data():
  first_data = True
  base_dir = 'archive'
  for filename in os.listdir(base_dir):
    stock_value = np.array(pd.read_csv(f'{base_dir}/{filename}')['VWAP'])
    stock_std = np.std(stock_value)
    stock_value = stock_value / stock_std
    daily_variance = np.array([np.log(stock_value[i]) - np.log(stock_value[i-1]) for i in range(1, len(stock_value))])
    realized_volatility = np.array([np.sqrt(np.sum(daily_variance[i-20:i]**2)) for i in range(20, len(daily_variance))])
    stock_value = stock_value[21:]
    data_len = len(stock_value)
    stock_value = np.array([stock_value[i-40:i] for i in list(range(40,data_len))])
    realized_volatility = np.array([realized_volatility[i-40:i] for i in list(range(40,data_len))])
    if first_data:
      data = np.stack([stock_value, realized_volatility], axis=2)
      first_data = False
    else:
      data = np.append(data, np.stack([stock_value, realized_volatility], axis=2), axis=0)

  data = tf.convert_to_tensor(data, dtype=tf.float32)
  data = tf.random.shuffle(data, seed=100)
  train_idx = 367334
  valid_idx = 367334 + 50000
  train = data[:train_idx]
  valid = data[train_idx:valid_idx]
  test = data[valid_idx:]

  return train, valid, test