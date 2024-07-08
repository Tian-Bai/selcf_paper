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
from utility import rf_config, rf_str, mlp_config, mlp_str, interaction_type, range_arg
import argparse

rf_param = ['n_estim', 'max_depth', 'max_features']
mlp_param = ['hidden', 'layers']

parser = argparse.ArgumentParser(description='Plot 4 targets (FDP, power, nsel and r^2) for any specified regressor and test case.')
parser.add_argument('-i', '--input', dest='itr', type=int, help='number of tests (seeds)', default=1000)
# parser.add_argument('-s', '--sigma', dest='sigma', type=str, help='sigma level', default='0.5(4)-0.2(4)')
parser.add_argument('-d', '--dim', dest='dim', type=int, help='number of features in generated data', default=20)
parser.add_argument('-n', '--ntest', dest='ntest', type=int, help='number of tests (m) in the setting', default=100)

subparsers = parser.add_subparsers(dest='regressor', required=True, help='The target regressor. Either "rf" or "mlp".')
parser_rf = subparsers.add_parser('rf', help='rf regressor parser.')
parser_mlp = subparsers.add_parser('mlp', help='mlp regressor parser.')

parser_rf.add_argument('xaxis', type=rf_str, help='x-axis in the plot')
parser_rf.add_argument('-r', '--range', type=range_arg, dest='range', help='range of the x-axis') # parsed as np.arange
parser_rf.add_argument('config', type=rf_config, help='other configurations of the rf') 

parser_mlp.add_argument('xaxis', type=mlp_str, help='x-axis in the plot')
parser_mlp.add_argument('-r', '--range', type=range_arg, dest='range', help='range of the x-axis') # parsed as np.arange
parser_mlp.add_argument('config', type=mlp_config, help='other configurations of the mlp')

args = parser.parse_args()

regressor = args.regressor
itr = args.itr
ntest = args.ntest
sigma = '0.5(4)-0.2(4)'
dim = args.dim
xaxis = args.xaxis
xrange = args.range
config = args.config

targets = [('power', 'Power'), ('nsel', 'Number of rejections')]

basic_gb = ['set', 'regressor', 'dim']

if regressor == 'rf':
    rf_param2 = [r for r in rf_param]
    rf_param2.remove(xaxis)
    df = pd.read_csv(f"..\\csv\\{regressor}\\{xaxis}\\{xrange[0]},{xrange[1]},{xrange[2]} {rf_param2[0]}={config[rf_param2[0]]} {rf_param2[1]}={config[rf_param2[1]]} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
    gb = basic_gb + rf_param # groupby parameters
elif regressor == 'mlp':
    mlp_param2 = [r for r in mlp_param]
    mlp_param2.remove(xaxis)
    df = pd.read_csv(f"..\\csv\\{regressor}\\{xaxis}\\{xrange[0]},{xrange[1]},{xrange[2]} {mlp_param2[0]}={config[mlp_param2[0]]} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
    gb = basic_gb + mlp_param

df = df.groupby(gb).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

grouped = df.groupby(['set'])

oracledf = pd.read_csv(f"..\\csv\\oracle\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
oracledf = oracledf.groupby(basic_gb).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

for (target, tname) in targets:
    fig, axs = plt.subplots(figsize=(20, 10), nrows = 2, ncols = 4, sharex=True, sharey=True)
    idx = 0

    for (s,), group in grouped:
        BH_res = []
        BH_rel = []
        BH_2clip = []
        bon = []
        r_sq = []
        for x in sorted(group[xaxis].unique()):
            BH_res.append(group[group[xaxis] == x][f'BH_res_{target}'].values[0])
            BH_rel.append(group[group[xaxis] == x][f'BH_rel_{target}'].values[0])
            BH_2clip.append(group[group[xaxis] == x][f'BH_2clip_{target}'].values[0])
            bon.append(group[group[xaxis] == x][f'Bonf_{target}'].values[0])
            r_sq.append(group[group[xaxis] == x][f'r_squared'].values[0])
        x = idx // 4
        y = idx % 4
        axs2 = axs[x][y].twinx()
        if idx == 0:
            # axs[x][y].plot(BH_res, marker='o', label="BH_res")
            axs[x][y].plot(np.arange(*xrange), BH_rel, label="BH_sub")
            axs[x][y].plot(np.arange(*xrange), BH_2clip, label="BH_2clip")
            axs2.plot(np.arange(*xrange), r_sq, color='green', linestyle='-', linewidth=5, alpha=0.6, label='R^2')

            axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'BH_rel_{target}'].values[0], linestyle='--', label='BH_sub (oracle)', alpha=0.8)
            axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'BH_2clip_{target}'].values[0], linestyle='--', label='BH_2clip (oracle)', alpha=0.8, color='orange')
            # axs[x][y].plot(bon, marker='o', label="Bonferroni")
        else:
            # axs[x][y].plot(BH_res, marker='o')
            axs[x][y].plot(np.arange(*xrange), BH_rel)
            axs[x][y].plot(np.arange(*xrange), BH_2clip)
            # axs[x][y].plot(bon, marker='o')
            axs2.plot(np.arange(*xrange), r_sq, color='green', linestyle='-', linewidth=5, alpha=0.6)

            axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'BH_rel_{target}'].values[0], linestyle='--', alpha=0.8)
            axs[x][y].axhline(y=oracledf[oracledf['set'] == s][f'BH_2clip_{target}'].values[0], linestyle='--', alpha=0.8, color='orange')
        axs[x][y].set_xlabel(f'Setting {s}')
        idx += 1
    fig.supxlabel(f"{regressor} - {xaxis}")
    fig.supylabel(f'{tname}')
    fig.suptitle(f"{tname} for different procedures and settings with control level 0.1, noise level {sigma} with {regressor} regressor. \n {ntest} tests and {dim} total features, averaged over {itr} times")
    fig.legend()
    plt.savefig(f'{regressor}-trendcomp {target} {xrange[0]},{xrange[1]},{xrange[2]} sigma={sigma} itr={itr} ntest={ntest} dim={dim}.png')