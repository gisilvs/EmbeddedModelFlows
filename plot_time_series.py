import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plot_utils import plot_data

rc = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"]}
plt.rcParams.update(rc)

model_name = 'lorenz_smoothing_r'
sp_name = 'normalizing_program_iaf'
rep_number = 0
with open(f'results_with_samples/{model_name}/{sp_name}/rep{rep_number}.pickle', 'rb') as handle:
  results = pickle.load(handle)

loss = results['loss']
elbo = results['elbo']
fkl = results['fkl']
ground_truth = results['ground_truth']
observations = results['observations']
samples = results['samples']

gs = gridspec.GridSpec(1, 3)

fig = plt.figure(figsize=[20,5])
ax = plt.subplot(gs[0,0])
plt.scatter(range(30), observations, c='black', alpha=0.5, label='Observations')
#plt.scatter(range(10), observations[:10], c='black', alpha=0.5)
#plt.scatter(range(20,30), observations[10:], c='black', alpha=0.5, label='Observations')
plt.plot(samples[:,:,0], c='orange', alpha = 0.25)
plt.plot(samples[:,0,0], label=f'Surrogate posterior', c='orange', alpha = 0.25)
plt.plot([g[0] for g in ground_truth], c='black')
plt.title('X')


handles, labels = ax.get_legend_handles_labels()
ax = plt.subplot(gs[0,1])
plt.plot(samples[:,:,1], c='orange', alpha = 0.25)
plt.plot([g[1] for g in ground_truth], c='black', label='Ground Truth')

h, l = ax.get_legend_handles_labels()
handles.extend(h)
labels.extend(l)
plt.title('Y')

ax = plt.subplot(gs[0,2])
plt.plot(samples[:,:,2], c='orange', alpha = 0.25)
plt.plot([g[2] for g in ground_truth], c='black')
h, l = ax.get_legend_handles_labels()
handles.extend(h)
labels.extend(l)
plt.title('Z')

#fig.legend(handles, labels, loc='upper right')
fig.suptitle('EMF-T', fontsize=16)
plt.savefig('emf_s.png')
