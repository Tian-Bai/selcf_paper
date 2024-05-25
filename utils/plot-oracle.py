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

df = pd.read_csv(f"..\\csv\\oracle.csv")
df = df[df['dim'] == 10]

df = df.groupby(['set', 'regressor']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])
df.to_csv("avg.csv")

grouped = df.groupby(['set'])
# bonf_grouped = average_bonf_df.groupby(['set', 'ntest'])

for (target, tname) in targets:
    fig, axs = plt.subplots(figsize=(20, 10), nrows = 2, ncols = 4, sharex=True, sharey=True)
    idx = 0

    for (s,), group in grouped:
        if target != 'r_squared':
            BH_res = group[f'BH_res_{target}'].values[0]
            BH_rel = group[f'BH_rel_{target}'].values[0]
            BH_2clip = group[f'BH_2clip_{target}'].values[0]
            bon = group[f'Bonf_{target}'].values[0]
        r_sq = group[f'r_squared'].values[0]

        x = idx // 4
        y = idx % 4
        if target != 'nsel':
            axs[x][y].set_ylim(top=1.1)
        if target != 'r_squared':
            if idx == 0:
                #axs[x][y].axhline(y=BH_res, color='red', label="BH_res")
                axs[x][y].axhline(y=BH_rel, label="BH_sub", alpha=0.8)
                axs[x][y].axhline(y=BH_2clip, color='orange', label="BH_2clip", alpha=0.8)
                #axs[x][y].axhline(y=bon, label="Bonferroni")
            else:
                #axs[x][y].axhline(y=BH_res)
                axs[x][y].axhline(y=BH_rel, alpha=0.8)
                axs[x][y].axhline(y=BH_2clip, color='orange', alpha=0.8)
                #axs[x][y].axhline(y=bon)
        else:
            axs[x][y].axhline(y=r_sq)
        
        axs[x][y].set_xlabel(f'Setting {s}')
        
        idx += 1

    # fig.text(0.38, 0.06, f"{t2} for different procedures, number of tests and settings")
    # fig.text(0.475, 0.08, "Noise level sigma")
    fig.supxlabel("For comparison")
    fig.supylabel(f'{tname}')
    fig.suptitle(f"{tname} for different procedures and settings with control level 0.1, noise level 0.5 (1-4) / 0.2 (5-8) and 100 tests with oracle regressor")
    fig.legend()
    plt.savefig(f'oracle performance {target}.png')