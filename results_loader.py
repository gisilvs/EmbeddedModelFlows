import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
import numpy as np

'''sps = ['mean_field',
       'multivariate_normal',
       'small_iaf',
       'large_iaf',
       'normalizing_program_mean_field',
       'normalizing_program_multivariate_normal',
       'normalizing_program_large_iaf',
       'normalizing_program_highway_flow']'''

sps = [
#'gated_normalizing_program_iaf',
  #'mean_field',
       #'multivariate_normal',
      #'asvi',
       # 'iaf',
       'normalizing_program_mean_field',
  #'normalizing_program_iaf',
  #'normalizing_program_multivariate_normal',

       #'normalizing_program_highway_flow',
       'gated_normalizing_program_mean_field',
       #'gated_normalizing_program_multivariate_normal',
  #'gated_normalizing_program_iaf',
       ]

root_dir='results'
results_dict = {}

for model in os.listdir(root_dir):
  if model == 'radon':
    continue
  if model == 'lorenz_smoothing_r':
    a = 0
  model_dir=f'{root_dir}/{model}'
  results_dict[model] = {}
  for surrogate_posterior in os.listdir(model_dir):
    if surrogate_posterior == 'normalizing_program_iaf':
      a = 0
    results_dict[model][surrogate_posterior]={}
    surrogate_posterior_dir = f'{model_dir}/{surrogate_posterior}'
    reps = []
    for rep in os.listdir(surrogate_posterior_dir):
      if 'checkpoints' in rep:
        continue
      try:
        with open(f'{surrogate_posterior_dir}/{rep}', 'rb') as handle:
          reps.append(pickle.load(handle))
      except:
        a = 0

    df = pd.DataFrame(reps)
    results_dict[model][surrogate_posterior]['elbo']= df.elbo
    results_dict[model][surrogate_posterior]['fkl'] = df.fkl

for k, models in results_dict.items():
    print(f'{k}')

    #bold_idx = np.argmin([models[s]["elbo"].mean() for s in sps])
    bold_idx = 100
    print("& -ELBO")
    for i, surrogate_posterior in enumerate(sps):
      f = ''
      if i == bold_idx:
        f += f' & $\\boldsymbol{{{models[surrogate_posterior]["elbo"].mean():.3f} \\pm {models[surrogate_posterior]["elbo"].sem():.3f}}}$'
      else:
        f += f' & ${models[surrogate_posterior]["elbo"].mean():.3f} \\pm {models[surrogate_posterior]["elbo"].sem():.3f}$'
      print(f)

    #bold_idx = np.argmax([models[s]["fkl"].mean() for s in sps])
    bold_idx=100
    print("& FKL")
    for i, surrogate_posterior in enumerate(sps):

      fkl_mean = models[surrogate_posterior]["fkl"].mean()
      fkl_sem = models[surrogate_posterior]["fkl"].sem()
      f = ''
      if i == bold_idx:
        if np.abs(fkl_mean) >= 1e4:
          fkl_mean = np.format_float_scientific(fkl_mean, precision=2)
          fkl_sem = np.format_float_scientific(fkl_sem, precision=2)
          f += f' & $\\boldsymbol{{{fkl_mean} \\pm {fkl_sem}}}$'
        else:
          f += f' & $\\boldsymbol{{{fkl_mean:.3f} \\pm {fkl_sem:.3f}}}$'
      else:
        if np.abs(fkl_mean) >= 1e4:
          fkl_mean = np.format_float_scientific(fkl_mean, precision=2)
          fkl_sem = np.format_float_scientific(fkl_sem, precision=2)
          f += f' & ${fkl_mean} \\pm {fkl_sem}$'
        else:
          f += f' & ${fkl_mean:.3f} \\pm {fkl_sem:.3f}$'
      print(f)


