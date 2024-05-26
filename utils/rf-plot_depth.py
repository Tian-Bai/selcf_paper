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
targets = [('nsel', 'Number of rejections'), ('r_squared', 'Out of sample R^2')]

df = pd.read_csv(f"..\\csv\\{regressor}-depth1000seed.csv")
df = df[df['dim'] == 10]

df = df.groupby(['set', 'regressor', 'n_estim', 'max_depth', 'max_features', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

oracledf = pd.read_csv(f"..\\csv\\oracle.csv")
oracledf = oracledf[oracledf['dim'] == 10]

oracledf = oracledf.groupby(['set', 'regressor']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

n_estim = 50 # 10, 20, ..., 100 or all
if n_estim != 'all':
    df = df[df['n_estim'] == n_estim]

grouped = df.groupby(['set'])
# bonf_grouped = average_bonf_df.groupby(['set', 'ntest'])

fig, axs = plt.subplots(figsize=(20, 10), nrows = 2, ncols = 4, sharex=True, sharey=True)

for (target, tname) in targets:
    # fig, axs = plt.subplots(figsize=(20, 10), nrows = 2, ncols = 4, sharex=True, sharey=True)
    idx = 0

    for (s,), group in grouped:
        BH_res = []
        BH_rel = []
        BH_2clip = []
        bon = []
        r_sq = []
        for depth in sorted(group['max_depth'].unique()):
            if depth > 20:
                continue
            if target != 'r_squared':
                BH_res.append(group[group['max_depth'] == depth][f'BH_res_{target}'].values[0])
                BH_rel.append(group[group['max_depth'] == depth][f'BH_rel_{target}'].values[0])
                BH_2clip.append(group[group['max_depth'] == depth][f'BH_2clip_{target}'].values[0])
                bon.append(group[group['max_depth'] == depth][f'Bonf_{target}'].values[0])
            else:
                r_sq.append(group[group['max_depth'] == depth][f'r_squared'].values[0])
        x = idx // 4
        y = idx % 4
        if target != 'r_squared':
            if idx == 0:
                # axs[x][y].plot(BH_res, marker='o', label="BH_res")
                axs[x][y].plot(BH_rel, label="BH_sub")
                axs[x][y].plot(BH_2clip, label="BH_2clip")
                # axs[x][y].plot(bon, marker='o', label="Bonferroni")
                axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'BH_rel_{target}'].values[0], linestyle='--', label="BH_sub (oracle)", alpha=0.8)
                axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'BH_2clip_{target}'].values[0], color='orange', linestyle='--', label="BH_2clip (oracle)", alpha=0.8)
            else:
                # axs[x][y].plot(BH_res, marker='o')
                axs[x][y].plot(BH_rel)
                axs[x][y].plot(BH_2clip)
                # axs[x][y].plot(bon, marker='o')
                axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'BH_rel_{target}'].values[0], linestyle='--', alpha=0.8)
                axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'BH_2clip_{target}'].values[0], color='orange', linestyle='--', alpha=0.8)
        else:
            axs2 = axs[x][y].twinx()
            # axs[x][y].plot(r_sq)
            # axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'r_squared'].values[0], linestyle='--', alpha=0.8)
            if idx == 0:
                axs2.plot(r_sq, color='green', linestyle='-.', alpha=0.6, label='R^2')
            else:
                axs2.plot(r_sq, color='green', linestyle='-.', alpha=0.6)
            # if idx == 0:
            #     axs2.axhline(y=oracledf[oracledf['set'] == s][f'r_squared'].values[0], linestyle='--', alpha=0.8, label='R^2')
            # else:
            #     axs2.axhline(y=oracledf[oracledf['set'] == s][f'r_squared'].values[0], linestyle='--', alpha=0.8)
        
        axs[x][y].set_xlabel(f'Setting {s}')
        
        idx += 1

    # fig.text(0.38, 0.06, f"{t2} for different procedures, number of tests and settings")
    # fig.text(0.475, 0.08, "Noise level sigma")
    fig.supxlabel("Max depth of the ranfom forest model")
    fig.supylabel(f'Number of rejections')
    fig.suptitle(f"{tname} and number of rejections for different procedures and settings with control level 0.1, noise level 0.5 and 100 tests with random forest regressor with {n_estim} trees. 10 total features")
    fig.legend()
plt.savefig(f'complexity-depth {target} {regressor} n_estim={n_estim}.png')