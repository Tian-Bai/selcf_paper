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
import argparse

# only focus on random forest (rf)
# only do this for setting 1, 2, 5, 6 where we have homogeneous variance

parser = argparse.ArgumentParser(description='Generate data for linear regressor experiment.')
parser.add_argument('-i', '--input', dest='itr', type=int, help='number of tests (seeds)', default=1000)
# parser.add_argument('-s', '--sigma', dest='sigma', type=str, help='sigma level', default='0.5(4)-0.2(4)')
parser.add_argument('-d', '--dim', dest='dim', type=int, help='number of features in generated data', default=20)
parser.add_argument('-n', '--ntest', dest='ntest', type=int, help='number of tests (m) in the setting', default=100)
parser.add_argument('--interaction', dest='interaction', type=bool, help='whether including interaction terms in the linear model', default=False)

args = parser.parse_args()

itr = args.itr
ntest = args.ntest
interaction = args.interaction
sigma = '0.5(4)-0.2(4)'
dim = args.dim
q = 0.1
spec = "interaction" if interaction else "simple"

targets = [('fdp', 'FDP'), ('power', 'Power'), ('nsel', 'Number of rejections'), ('r_squared', 'Out of sample R^2')] # 'power', 'nsel'

df = pd.read_csv(f"..\\csv\\linear\\{spec}\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")

df = df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

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
    fig.suptitle(f"{tname} for different procedures and settings with control level 0.1, noise level {sigma} with linear-{spec} regressor. \n {ntest} tests and {dim} total features, averaged over {itr} times.")
    fig.legend()
    plt.savefig(f'linear-{spec} {target} sigma={sigma} itr={itr} ntest={ntest} dim={dim}.png')