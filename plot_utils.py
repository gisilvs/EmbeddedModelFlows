import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_data(model_name, ground_truth, observations, samples=[]):
  plt.style.use('seaborn')
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
    plt.plot([s[:,0] for s in samples])
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

def plot_heatmap_2d(dist, matching_bijector=None, xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0,
                    mesh_count=1000, name=None):

  fig = plt.figure(frameon=False)

  x = tf.linspace(xmin, xmax, mesh_count)
  y = tf.linspace(ymin, ymax, mesh_count)
  X, Y = tf.meshgrid(x, y)

  concatenated_mesh_coordinates = tf.transpose(
    tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])]))
  if matching_bijector:
    concatenated_mesh_coordinates = matching_bijector(concatenated_mesh_coordinates)
  prob = dist.prob(concatenated_mesh_coordinates)
  # plt.hexbin(concatenated_mesh_coordinates[:,0], concatenated_mesh_coordinates[:,1], C=prob, cmap='rainbow')
  prob = prob.numpy()

  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(tf.transpose(tf.reshape(prob, (mesh_count, mesh_count))),
             aspect="equal")
  if name:
    fig.savefig(name, format="png")

def plot_samples(samples, npts=1000, low=-4, high=4, name=None):
  fig = plt.figure(frameon=False)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.hist2d(samples[:, 0], samples[:, 1], range=[[low, high], [low, high]],
            bins=npts)
  ax.invert_yaxis()
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([])
  ax.set_aspect('equal')
  fig.add_axes(ax)

  if name:
    fig.savefig(name, format="png")