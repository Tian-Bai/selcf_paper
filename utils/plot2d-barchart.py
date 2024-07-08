import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import random
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from utility import gen_data, BH
import argparse

# plot the barchart for linear, additive, DL and Rf

parser = argparse.ArgumentParser(description='Experiment specifications.')
parser.add_argument('-i', '--input', dest='itr', type=int, help='number of tests (seeds)', default=1000)
# parser.add_argument('-s', '--sigma', dest='sigma', type=str, help='sigma level', default='0.5(4)-0.2(4)')
parser.add_argument('-d', '--dim', dest='dim', type=int, help='number of features in generated data', default=20)
parser.add_argument('-n', '--ntest', dest='ntest', type=int, help='number of tests (m) in the setting', default=100)

args = parser.parse_args()

itr = args.itr
ntest = args.ntest
sigma = '0.5(4)-0.2(4)'
dim = args.dim
q = 0.1

targets = [('fdp', 'FDP'), ('power', 'Power'), ('nsel', 'Number of rejections'), ('r_squared', 'Out of sample R^2')] # 'power', 'nsel'

linear_no_df = pd.read_csv(f"..\\csv2d\\linear\\interaction=no\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
linear_oracle_df = pd.read_csv(f"..\\csv2d\\linear\\interaction=oracle\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
# linear_yes_df = pd.read_csv(f"..\\csv\\linear\\interaction=yes\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")

# additive_no_df = pd.read_csv(f"..\\csv\\additive\\interaction=no\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
# additive_oracle_df = pd.read_csv(f"..\\csv\\additive\\interaction=oracle\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
# additive_yes_df = pd.read_csv(f"..\\csv\\additive\\interaction=yes\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")

linear_no_df['regressor'] = 'linear-no'
linear_no_df = linear_no_df.drop(columns=['interaction'])
linear_oracle_df['regressor'] = 'linear-oracle'
linear_oracle_df = linear_oracle_df.drop(columns=['interaction'])

# additive_no_df['regressor'] = 'additive-no'
# additive_no_df = additive_no_df.drop(columns=['interaction'])
# additive_oracle_df['regressor'] = 'additive-oracle'
# additive_oracle_df = additive_oracle_df.drop(columns=['interaction'])

rf_df = pd.read_csv(f"..\\csv2d\\rf\\max_depth\\20,21,1 n_estim=50 max_features=10 ntest={ntest} itr={itr} sigma={sigma} dim=20.csv")
rf_df = rf_df[rf_df["max_depth"] == 20]
rf_df = rf_df.drop(columns=['max_features', 'max_depth', 'n_estim'])

mlp_df = pd.read_csv(f"..\\csv2d\\mlp\\hidden\\32,33,1 layers=3 ntest={ntest} itr={itr} sigma={sigma} dim=20.csv")
mlp_df = mlp_df[mlp_df["layers"] == 3]
mlp_df = mlp_df.drop(columns=['hidden', 'layers'])

linear_no_df = linear_no_df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])
linear_oracle_df = linear_oracle_df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])
# linear_yes_df = linear_yes_df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

# additive_no_df = additive_no_df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])
# additive_oracle_df = additive_oracle_df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])
# additive_yes_df = additive_yes_df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

rf_df = rf_df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])
mlp_df = mlp_df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

combined_df = pd.concat([linear_no_df, linear_oracle_df, rf_df, mlp_df], axis=0, ignore_index=True)
# combined_df = pd.concat([linear_no_df, linear_oracle_df, additive_no_df, additive_oracle_df, rf_df, mlp_df], axis=0, ignore_index=True)

oracledf = pd.read_csv(f"..\\csv2d\\oracle\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
oracledf = oracledf.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

grouped = combined_df.groupby(['set'])
# bonf_grouped = average_bonf_df.groupby(['set', 'ntest'])

regressor = ['linear-no', 'linear-oracle', 'rf', 'mlp']
categories_name = ['linear', 'lin.-inter.', 'rf', 'mlp']

for (target, tname) in targets:
    fig, axs = plt.subplots(figsize=(14, 10), nrows = 2, ncols = 2, sharex=True, sharey=True)
    idx = 0

    for (s,), group in grouped:
        if s not in [1, 2, 5, 6]:
            continue
        BH_rel = []
        BH_2clip = []
        r_sq = []
        for reg in regressor:
            if target != 'r_squared':
                BH_rel.append(group[group["regressor"] == reg][f'BH_rel_{target}'].values[0])
                BH_2clip.append(group[group["regressor"] == reg][f'BH_2clip_{target}'].values[0])
            else:
                r_sq.append(group[group["regressor"] == reg][f'r_squared'].values[0])
        x = idx // 2
        y = idx % 2
        if target != 'r_squared':
            if idx == 0:
                axs[x][y].bar(np.arange(len(categories_name)), BH_rel, 0.2, label='BH_sub')
                axs[x][y].bar(np.arange(len(categories_name)) + 0.2, BH_2clip, 0.2, label='BH_2clip')
                if target != 'fdp':
                    axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'BH_rel_{target}'].values[0], linestyle='--', label='BH_sub (oracle)', alpha=0.8)
                    axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'BH_2clip_{target}'].values[0], linestyle='--', label='BH_2clip (oracle)', alpha=0.8, color='orange')
            else:
                axs[x][y].bar(np.arange(len(categories_name)), BH_rel, 0.2)
                axs[x][y].bar(np.arange(len(categories_name)) + 0.2, BH_2clip, 0.2)
                if target != 'fdp':
                    axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'BH_rel_{target}'].values[0], linestyle='--', alpha=0.8)
                    axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'BH_2clip_{target}'].values[0], linestyle='--', alpha=0.8, color='orange')
            axs[x][y].set_xticks(np.arange(len(categories_name)) + 0.1, categories_name)
            if target == 'fdp':
                axs[x][y].axhline(0.1, color='green')
        else:
            if idx == 0:
                axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'{target}'].values[0], linestyle='--', label='oracle', alpha=0.8)
            else:
                axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'{target}'].values[0], linestyle='--', alpha=0.8)
            axs[x][y].bar(categories_name, r_sq, 0.35)
            axs[x][y].axhline(0, color='black')
        
        axs[x][y].set_xlabel(f'Setting {s if s < 5 else s - 2}')
        
        idx += 1

    # fig.text(0.38, 0.06, f"{t2} for different procedures, number of tests and settings")
    # fig.text(0.475, 0.08, "Noise level sigma")
    fig.supxlabel("For comparison")
    fig.supylabel(f'{tname}')
    fig.suptitle(f"{tname} for different procedures and settings with control level 0.1, noise level {sigma}. \n {ntest} tests and {dim} total features, averaged over {itr} times.")
    fig.legend()
    plt.savefig(f'barchart {target} sigma={sigma} itr={itr} ntest={ntest} dim={dim}.png')