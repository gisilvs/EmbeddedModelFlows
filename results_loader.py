import os
import pickle
import pandas as pd
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

sps = ['mean_field',
       'multivariate_normal',
       'asvi',
       'iaf',
       'normalizing_program_mean_field',
       'normalizing_program_multivariate_normal',
       'normalizing_program_iaf',
       'normalizing_program_highway_flow',
       ]

root_dir='results'
results_dict = {}

for model in os.listdir(root_dir):
  if model == 'lorenz_bridge_r':
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
      with open(f'{surrogate_posterior_dir}/{rep}', 'rb') as handle:
        reps.append(pickle.load(handle))

    df = pd.DataFrame(reps)
    results_dict[model][surrogate_posterior]['elbo']= df.elbo
    results_dict[model][surrogate_posterior]['fkl'] = df.fkl

for k, models in results_dict.items():
    print(f'{k}')

    print("& ELBO")
    for surrogate_posterior in sps:
      f = ''
      f += f' & ${models[surrogate_posterior]["elbo"].mean():.3f} \\pm {models[surrogate_posterior]["elbo"].sem():.3f}$'
      print(f)

    print("& FKL")
    for surrogate_posterior in sps:
      f = ''
      f += f' & ${models[surrogate_posterior]["fkl"].mean():.3f} \\pm {models[surrogate_posterior]["fkl"].sem():.3f}$'
      print(f)


