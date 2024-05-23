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

regressor = 'rf' # 'rf', 'gbr', 'svm'
targets = [('fdp', 'FDP'), ('power', 'Power'), ('nsel', 'Number of rejections'), ('r_squared', 'Out of sample R^2')] # 'power', 'nsel'

df = pd.read_csv(f"..\\csv\\{regressor}-features-0.5v0.2.csv")
df = df.groupby(['set', 'regressor', 'n_estim', 'max_depth', 'max_features']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])
df.to_csv("avg.csv")

n_estim = 50
if n_estim != 'all':
    df = df[df['n_estim'] == n_estim]

grouped = df.groupby(['set'])
# bonf_grouped = average_bonf_df.groupby(['set', 'ntest'])

for (target, tname) in targets:
    fig, axs = plt.subplots(figsize=(20, 10), nrows = 2, ncols = 4, sharex=True, sharey=True)
    idx = 0

    for (s,), group in grouped:
        BH_res = []
        BH_rel = []
        BH_2clip = []
        bon = []
        r_sq = []
        for features in sorted(group['max_features'].unique()):
            if target != 'r_squared':
                BH_res.append(group[group['max_features'] == features][f'BH_res_{target}'].values[0])
                BH_rel.append(group[group['max_features'] == features][f'BH_rel_{target}'].values[0])
                BH_2clip.append(group[group['max_features'] == features][f'BH_2clip_{target}'].values[0])
                bon.append(group[group['max_features'] == features][f'Bonf_{target}'].values[0])
            else:
                r_sq.append(group[group['max_features'] == features][f'r_squared'].values[0])
        x = idx // 4
        y = idx % 4
        if target != 'r_squared':
            if idx == 0:
                axs[x][y].plot(BH_res, marker='o', label="BH_res")
                axs[x][y].plot(BH_rel, marker='o', label="BH_sub")
                axs[x][y].plot(BH_2clip, marker='o', label="BH_2clip")
                axs[x][y].plot(bon, marker='o', label="Bonferroni")
            else:
                axs[x][y].plot(BH_res, marker='o')
                axs[x][y].plot(BH_rel, marker='o')
                axs[x][y].plot(BH_2clip, marker='o')
                axs[x][y].plot(bon, marker='o')
        else:
            axs[x][y].plot(r_sq, marker='o')
        
        axs[x][y].set_xlabel(f'Setting {s}')
        
        idx += 1

    # fig.text(0.38, 0.06, f"{t2} for different procedures, number of tests and settings")
    # fig.text(0.475, 0.08, "Noise level sigma")
    fig.supxlabel("Max number of features of the ranfom forest model")
    fig.supylabel(f'{tname}')
    fig.suptitle(f"{tname} for different procedures and settings with control level 0.1, noise level 0.5 (1-4) / 0.2 (5-8) and 100 tests with random forest regressor with {n_estim} trees")
    fig.legend()
    plt.savefig(f'complexity-features {target} {regressor} n_estim={n_estim}.png')