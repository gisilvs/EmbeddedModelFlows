import pickle
import os
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

names = {
  'maf': 'MAF',
  'maf3': 'MAF-L',
  'np_maf': 'GEMF-T',
  'c100_np_maf': 'GEMF-T',
  'sandwich': 'GEMF-M',
  'c100_sandwich': 'GEMF-M',
  'np_maf_smoothness': 'GEMF-T(s)',
  'bottom': 'B-MAF'
}

d_names = {
  'brownian': 'Brownian motion',
  'ornstein': 'Ornstein-Uhlenbeck process',
  'lorenz': 'Lorenz system'
}

base_dir = 'hierarchical_results'

if base_dir == '2d_toy_results':
  datasets = ['8gaussians', 'checkerboard']
  models = ['c100_np_maf', 'c100_sandwich', 'maf', 'maf3']

elif base_dir == 'time_series_results':
  datasets = ['brownian', 'ornstein', 'lorenz']
  models = ['np_maf_continuity', 'np_maf_smoothness', 'maf','maf3','bottom']

elif base_dir == 'hierarchical_results':
  datasets = ['digits']
  models = ['np_maf', 'sandwich', 'maf','maf3']

results = {}
for dataset in datasets:
  results[dataset] = {}
  for model in models:
    results[dataset][model] = []
for run in os.listdir(base_dir):
  if 'run' in run:
    if datasets:
      for dataset in datasets:
        for model in models:
          try:
            with open(f'{base_dir}/{run}/{dataset}/{model}.pickle', 'rb') as handle:
              res = pickle.load(handle)
          except:
            a = 0
          # plt.plot(res['loss'], label=names[model])
          results[dataset][model].append(res['loss_eval'])
        '''if dataset == 'lorenz':
          plt.ylim(bottom=-300, top=800)
        plt.title('Digits')
        plt.legend()
        plt.savefig(f'hierarchical_results/loss_{dataset}.png')
        plt.close()'''


for d, dataset in results.items():
    print(f'{d}')

    bold_idx = np.argmin([np.mean(dataset[m]) for m in models])
    print("& -LOGP")
    for i, model in enumerate(models):
      f = ''
      if i == bold_idx:
        f += f' & $\\boldsymbol{{{np.mean(dataset[model]):.3f} \\pm {sem(dataset[model]):.4f}}}$'
      else:
        f += f' & ${np.mean(dataset[model]):.3f} \\pm {sem(dataset[model]):.4f}$'
      print(f)

