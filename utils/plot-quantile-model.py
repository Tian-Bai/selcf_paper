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

rf_param = ['n_estim', 'max_depth', 'max_features']
mlp_param = ['hidden', 'layers']

''' 
If the regressor is rf, parameters are ['n_estim', 'max_depth', 'max_features'].
If the regressor if mlp, parameters are ['hidden', 'layers'].
'''

def range_arg(value):
    try:
        values = [float(i) for i in value.split(',')]
        assert len(values) == 3
        values[2] = int(values[2])
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            'Provide a comma-seperated list of 1, 2 or 3 integers'
        )
    return values

def rf_config(value):
    try:
        pairs = {}
        for pair in value.split(','):
            k, v = pair.split(':')
            assert k in rf_param
            pairs[k.strip()] = v.strip()
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            'Illegal argument for rf configurations.'
        )
    return pairs

def mlp_config(value):
    try:
        pairs = {}
        for pair in value.split(','):
            k, v = pair.split(':')
            assert k in mlp_param
            pairs[k.strip()] = v.strip()
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            'Illegal argument for mlp configurations.'
        )
    return pairs

def interaction_type(value):
    try:
        s = str(value).lower()
        assert s in ['yes', 'y', 'no', 'n', 'oracle', 'o']
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            'Illegal argument for linear model type. Should be either "yes", "no", or "oracle".'
        )
    if s == 'y':
        s = 'yes'
    elif s == 'n':
        s = 'no'
    elif s == 'o':
        s = 'oracle'
    return s

# parsers, and general configurations
parser = argparse.ArgumentParser(description='Generate data for 4 targets (FDP, power, nsel and r^2) for any specified regressor and test case.')
parser.add_argument('-i', '--input', dest='itr', type=int, help='number of tests (seeds)', default=1000)
# parser.add_argument('-s', '--sigma', dest='sigma', type=str, help='sigma level', default='0.5(4)-0.2(4)')
parser.add_argument('-d', '--dim', dest='dim', type=int, help='number of features in generated data', default=20)
parser.add_argument('-n', '--ntest', dest='ntest', type=int, help='number of tests (m) in the setting', default=100)
parser.add_argument('-qt', '--quantile', dest='qt', type=range_arg, help='quantile range to consider')

# subparsers for different supported models
subparsers = parser.add_subparsers(dest='regressor', required=True, help='The target regressor. Choose between ["rf", "mlp", "additive", "linear", ...].')
parser_rf = subparsers.add_parser('rf', help='rf regressor parser.')
parser_mlp = subparsers.add_parser('mlp', help='mlp regressor parser.')
parser_linear = subparsers.add_parser('linear', help='linear regressor parser.')
parser_additive = subparsers.add_parser('additive', help='GAM regressor parser.')

# rf parser
parser_rf.add_argument('config', type=rf_config, help='other configurations of the rf') 

# mlp parser
parser_mlp.add_argument('config', type=mlp_config, help='other configurations of the mlp')

# for below two regressors, linear and additive, we allow choosing between whether to use interaction between the terms, and whether the interaction terms are 'oracle'.
# linear parser
parser_linear.add_argument('--interaction', dest='interaction', type=interaction_type, help='whether including interaction terms in the linear model', default=False)

# additive parser
parser_additive.add_argument('--interaction', dest='interaction', type=interaction_type, help='whether including interaction terms in the additive model', default=False)

args = parser.parse_args()

regressor = args.regressor
itr = args.itr
ntest = args.ntest
sigma = '0.5(4)-0.2(4)'
dim = args.dim
q = 0.1

# quantiles to consider
qt_list = args.qt

targets = [('fdp', 'FDP'), ('power', 'Power'), ('nsel', 'Number of rejections')]
# r_sq is meaningless for quantile regression

if regressor == 'rf':
    config = args.config

    df = pd.read_csv(f"..\\csv\\quantile-{regressor}\\{qt_list[0]},{qt_list[1]},{qt_list[2]} {rf_param[0]}={config[rf_param[0]]} {rf_param[1]}={config[rf_param[1]]} {rf_param[2]}={config[rf_param[2]]} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
    gb = ['set', 'regressor', 'dim', 'qt']
elif regressor == 'mlp':
    config = args.config

    df = pd.read_csv(f"..\\csv\\quantile-{regressor}\\{qt_list[0]},{qt_list[1]},{qt_list[2]} {mlp_param[0]}={config[mlp_param[0]]} {mlp_param[1]}={config[mlp_param[1]]} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
    gb = ['set', 'regressor', 'dim', 'qt']
elif regressor in ['linear', 'additive']:
    interaction = args.interaction

    df = pd.read_csv(f"..\\csv\\quantile-{regressor}\\{qt_list[0]},{qt_list[1]},{qt_list[2]} interaction={interaction} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
    gb = ['set', 'regressor', 'dim', 'qt']

df = df.groupby(gb).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

grouped = df.groupby(['set'])

for (target, tname) in targets:
    fig, axs = plt.subplots(figsize=(20, 10), nrows = 2, ncols = 4, sharex=True, sharey=True)
    idx = 0

    for (s,), group in grouped:
        BH_res = []
        BH_rel = []
        BH_2clip = []
        bon = []
        r_sq = []
        qt_range = sorted(group['qt'].unique())
        for iqt in qt_range:
            if target != 'r_squared':
                BH_res.append(group[(group['qt'] == iqt)][f'BH_res_{target}'].values[0])
                BH_rel.append(group[(group['qt'] == iqt)][f'BH_rel_{target}'].values[0])
                BH_2clip.append(group[(group['qt'] == iqt)][f'BH_2clip_{target}'].values[0])
                bon.append(group[(group['qt'] == iqt)][f'Bonf_{target}'].values[0])
            else:
                r_sq.append(group[(group['qt'] == iqt)][f'r_squared'].values[0])
        x = idx // 4
        y = idx % 4
        if target != 'r_squared':
            if idx == 0:
                #axs[x][y].axhline(y=BH_res, color='red', label="BH_res")
                axs[x][y].plot(qt_range, BH_rel, label="BH_sub")
                axs[x][y].plot(qt_range, BH_2clip, label="BH_2clip")
                #axs[x][y].axhline(y=bon, label="Bonferroni")
            else:
                #axs[x][y].axhline(y=BH_res)
                axs[x][y].plot(qt_range, BH_rel)
                axs[x][y].plot(qt_range, BH_2clip)
                #axs[x][y].axhline(y=bon)
        else:
            pass
        
        axs[x][y].set_xlabel(f'Setting {s}')
        idx += 1

    fig.supxlabel(f"quantile-{regressor} - qt")
    fig.supylabel(f'{tname}')
    fig.suptitle(f"{tname} for different procedures and settings with control level 0.1, noise level {sigma} with quantile-{regressor} regressor. \n {ntest} tests and {dim} total features, averged over {itr} times.")
    fig.legend()
    if regressor == 'rf':
        plt.savefig(f'quantile-{regressor} {target} {qt_list[0]},{qt_list[1]},{qt_list[2]} {rf_param[0]}={config[rf_param[0]]} {rf_param[1]}={config[rf_param[1]]} {rf_param[2]}={config[rf_param[2]]} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.png')
    elif regressor == 'mlp':
        plt.savefig(f'quantile-{regressor} {target} {qt_list[0]},{qt_list[1]},{qt_list[2]} {mlp_param[0]}={config[mlp_param[0]]} {mlp_param[1]}={config[mlp_param[1]]} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv')
    elif regressor in ['linear', 'additive']:
        plt.savefig(f'quantile-{regressor} {target} {qt_list[0]},{qt_list[1]},{qt_list[2]} interaction={interaction} sigma={sigma} itr={itr} ntest={ntest} dim={dim}.png')