import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

def plot_data(model_name, ground_truth, observations):
  if model_name == 'brownian_bridge_r' or model_name == 'brownian_bridge_c':
    plt.plot(ground_truth)
    plt.scatter(range(10),observations[:10], c='g')
    plt.scatter(range(20,30),observations[10:], c='g')

  elif model_name == 'brownian_smoothing_r' or model_name == 'brownian_smoothing_c':
    plt.plot(ground_truth)
    plt.scatter(range(30), observations, c='g')

  elif model_name == 'lorenz_bridge_r':
    plt.plot([g[0] for g in ground_truth])
    plt.plot([g[1] for g in ground_truth])
    plt.plot([g[2] for g in ground_truth])
    plt.scatter(range(10), observations[:10], c='g')
    plt.scatter(range(20, 30), observations[10:], c='g')

  elif model_name == 'lorenz_bridge_c':
    plt.plot([g[0] for g in ground_truth])
    plt.plot([g[1] for g in ground_truth])
    plt.plot([g[2] for g in ground_truth])
    plt.scatter(range(10), 20*np.array(observations[:10]), c='g')
    plt.scatter(range(20, 30), 20*np.array(observations[10:]), c='g')

  elif model_name == 'lorenz_smoothing_r':
    plt.plot([g[0] for g in ground_truth])
    plt.plot([g[1] for g in ground_truth])
    plt.plot([g[2] for g in ground_truth])
    plt.scatter(range(30), observations, c='g')

  elif model_name == 'lorenz_smoothing_c':
    plt.plot([g[0] for g in ground_truth])
    plt.plot([g[1] for g in ground_truth])
    plt.plot([g[2] for g in ground_truth])
    plt.scatter(range(30), 20*np.array(observations), c='g')

  plt.show()