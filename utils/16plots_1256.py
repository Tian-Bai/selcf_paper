import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import random
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from utils import gen_data, BH

# only focus on random forest (rf)
# only do this for setting 1, 2, 5, 6 where we have homogeneous variance

''' 
1. generate the FDR and power plot as in the paper
'''

target = 'power' # 'fdp', 'power'
if target == 'power':
    t2 = 'Power'
elif target == 'fdp':
    t2 = 'FDP'

df = pd.read_csv("..\\results\\all.csv")
df = df.groupby(['sigma', 'q', 'set', 'ntest', 'regressor']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

df.to_csv("avg.csv")

fig, axs = plt.subplots(figsize=(10, 10), nrows = 4, ncols = 4, sharex=True, sharey=True)

idx = 0
grouped = df.groupby(['set', 'ntest'])
# bonf_grouped = average_bonf_df.groupby(['set', 'ntest'])

for (s, n), group in grouped:
    BH_res = []
    BH_rel = []
    BH_2clip = []
    bon = []
    for sigma in sorted(group['sigma'].unique()):
        BH_res.append(group[group['sigma'] == sigma][f'BH_res_{target}'].values[0])
        BH_rel.append(group[group['sigma'] == sigma][f'BH_rel_{target}'].values[0])
        BH_2clip.append(group[group['sigma'] == sigma][f'BH_2clip_{target}'].values[0])
        bon.append(group[group['sigma'] == sigma][f'Bonf_{target}'].values[0])
    x = idx // 4
    y = idx % 4
    if idx == 0:
        axs[x][y].plot(BH_res, marker='o', label="BH_res")
        axs[x][y].plot(BH_rel, marker='o', label="BH_rel")
        axs[x][y].plot(BH_2clip, marker='o', label="BH_2clip")
        axs[x][y].plot(bon, marker='o', label="Bonferroni")
    else:
        axs[x][y].plot(BH_res, marker='o')
        axs[x][y].plot(BH_rel, marker='o')
        axs[x][y].plot(BH_2clip, marker='o')
        axs[x][y].plot(bon, marker='o')
    
    if y % 4 == 3: # right
        axs[x][y].yaxis.set_label_position('right')
        #axs[x][y].yaxis.set_ticks_position('right')
        axs[x][y].set_ylabel(f'Setting {s}')
    
    if x == 0: # top
        axs[x][y].xaxis.set_label_position('top')
        axs[x][y].set_xlabel(f'ntest = {n}')
    idx += 1

# fig.text(0.38, 0.06, f"{t2} for different procedures, number of tests and settings")
# fig.text(0.475, 0.08, "Noise level sigma")
fig.supxlabel("Noise level sigma")
fig.supylabel(f'{t2}')
fig.suptitle(f"{t2} for different procedures, number of tests and settings with control level 0.1")
fig.legend()

plt.savefig(f'{target}.png')