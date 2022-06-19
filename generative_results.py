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
  'c100_sandwich_bn': 'GEMF-M',
  'np_maf_smoothness': 'GEMF-T(s)',
  'np_maf_continuity': 'GEMF-T(c)',
  'bottom': 'B-MAF',
  'splines': 'NSF',
  'c100_np_splines': 'NSF-EMF-T',
  'c100_sandwich_splines': 'NSF-EMF-M',
  'c100_sandwich_splines_bn': 'NSF-EMF-M',
  'np_splines_continuity': 'NSF-EMF-T(c)',
}

d_names = {
  'brownian': 'Brownian motion',
  'ornstein': 'Ornstein-Uhlenbeck process',
  'lorenz': 'Lorenz system',
  'mnist': 'MNIST',
  'van_der_pol': "Van der Pol oscillator"
}

base_dir = 'all_results/time_series_results'

if base_dir == 'mnist':
  datasets = ['mnist']
  models = [
    'c100_np_maf',
    'c100_sandwich_bn',
    'c100_np_splines',
    'c100_sandwich_splines_bn',
    'maf',
    'maf3',
    'splines'
  ]#
  # 'c100_sandwich', 'maf_bn',
            #'splines_bn', 'c100_np_maf_bn', 'c10_np_maf']

if base_dir == '2d_toy_results':
  datasets = ['checkerboard', '8gaussians']
  models = ['c100_np_maf', 'c100_sandwich', 'c100_np_splines','c100_sandwich_splines', 'maf', 'maf3', 'splines']

elif base_dir == 'time_series_results':
  datasets = ['van_der_pol'] # , 'ornstein']
  models = ['np_maf_continuity', 'np_splines_continuity','maf', 'maf3','splines'
            ,'bottom']

elif base_dir == 'hierarchical_results':
  datasets = ['digits']
  models = ['np_maf', 'sandwich', 'maf','maf3']

results = {}
for dataset in datasets:
  results[dataset] = {}
  for model in models:
    results[dataset][model] = []
for run in os.listdir(base_dir):
  if 'run_0' in run:
    if datasets:
      for dataset in datasets:
        for model in models:
          try:
            #with open(f'{base_dir}/{run}/{dataset}/{model}.pickle', 'rb') as
            # handle:
            if base_dir =='mnist':
              full_path = f'{base_dir}/{run}/{model}.pickle'
            else:
              full_path = f'{base_dir}/{run}/{dataset}/{model}.pickle'
            with open(full_path, 'rb') as \
                handle:
              res = pickle.load(handle)
              results[dataset][model].append(res['loss_eval'])
          except:
            a = 0
          plt.plot(res['loss'], label=names[model], alpha=0.9)
        # if dataset == 'lorenz':
        # plt.ylim(bottom=-250, top=100)
        plt.ylim(top=0)
        plt.title(f'{d_names[dataset]}')
        plt.legend()
        plt.savefig(f'{base_dir}/loss_{dataset}.png')
        plt.close()


for d, dataset in results.items():
    print(f'{d}')
    bold_idx = np.argmin([np.mean(dataset[m]) for m in models])
    print("& -LOGP")
    for i, model in enumerate(models):
      f = ''
      if i == bold_idx:
        f += f' & $\\boldsymbol{{{np.mean(dataset[model]):.3f} \\pm {sem(dataset[model]):.3f}}}$'
      else:
        f += f' & ${np.mean(dataset[model]):.3f} \\pm {sem(dataset[model]):.3f}$'
      print(f)

