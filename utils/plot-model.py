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
        values = [int(i) for i in value.split(',')]
        assert len(values) == 3
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            'Provide a comma-seperated list of 1, 2 or 3 integers'
        )
    return values

def rf_str(value):
    try:
        assert str(value) in rf_param
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            'Illegal argument for rf x-axis.'
        )
    return str(value)

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

def mlp_str(value):
    try:
        assert str(value) in mlp_param
    except(ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            'Illegal argument for mlp x-axis.'
        )
    return str(value)

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
parser = argparse.ArgumentParser(description='Plot 4 targets (FDP, power, nsel and r^2) for any specified regressor and test case.')
parser.add_argument('-i', '--input', dest='itr', type=int, help='number of tests (seeds)', default=1000)
# parser.add_argument('-s', '--sigma', dest='sigma', type=str, help='sigma level', default='0.5(4)-0.2(4)')
parser.add_argument('-d', '--dim', dest='dim', type=int, help='number of features in generated data', default=20)
parser.add_argument('-n', '--ntest', dest='ntest', type=int, help='number of tests (m) in the setting', default=100)

# subparsers for different supported models
subparsers = parser.add_subparsers(dest='regressor', required=True, help='The target regressor. Choose between ["rf", "mlp", "additive", "linear", ...].')
parser_rf = subparsers.add_parser('rf', help='rf regressor parser.')
parser_mlp = subparsers.add_parser('mlp', help='mlp regressor parser.')
parser_linear = subparsers.add_parser('linear', help='linear regressor parser.')
parser_additive = subparsers.add_parser('additive', help='GAM regressor parser.')

# for below two regressors, rf and mlp, we allow testing along an x axis representing the configuration of models (e.g. number of hidden layers, ...)
# rf parser
parser_rf.add_argument('xaxis', type=rf_str, help='x-axis in the plot')
parser_rf.add_argument('-r', '--range', type=range_arg, dest='range', help='range of the x-axis') # parsed as np.arange
parser_rf.add_argument('config', type=rf_config, help='other configurations of the rf') 

# mlp parser
parser_mlp.add_argument('xaxis', type=mlp_str, help='x-axis in the plot')
parser_mlp.add_argument('-r', '--range', type=range_arg, dest='range', help='range of the x-axis') # parsed as np.arange
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

targets = [('fdp', 'FDP'), ('power', 'Power'), ('nsel', 'Number of rejections'), ('r_squared', 'Out of sample R^2')]

if regressor == 'rf':
    xaxis = args.xaxis
    xrange = args.range
    config = args.config

    rf_param2 = [r for r in rf_param]
    rf_param2.remove(xaxis)
    df = pd.read_csv(f"..\\csv\\{regressor}\\{xaxis}\\{xrange[0]},{xrange[1]},{xrange[2]} {rf_param2[0]}={config[rf_param2[0]]} {rf_param2[1]}={config[rf_param2[1]]} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
    gb = ['set', 'regressor', 'dim'] + rf_param
elif regressor == 'mlp':
    xaxis = args.xaxis
    xrange = args.range
    config = args.config

    mlp_param2 = [r for r in mlp_param]
    mlp_param2.remove(xaxis)
    df = pd.read_csv(f"..\\csv\\{regressor}\\{xaxis}\\{xrange[0]},{xrange[1]},{xrange[2]} {mlp_param2[0]}={config[mlp_param2[0]]} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
    gb = ['set', 'regressor', 'dim'] + mlp_param
elif regressor in ['linear', 'additive']:
    interaction = args.interaction

    df = pd.read_csv(f"..\\csv\\{regressor}\\interaction={interaction}\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
    gb = ['set', 'regressor', 'dim', 'interaction']

df = df.groupby(gb).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

grouped = df.groupby(['set'])

for (target, tname) in targets:
    fig, axs = plt.subplots(figsize=(20, 10), nrows = 2, ncols = 4, sharex=True, sharey=True)
    idx = 0

    for (s,), group in grouped:
        if regressor in ['mlp', 'rf']:
            BH_res = []
            BH_rel = []
            BH_2clip = []
            bon = []
            r_sq = []
            for x in sorted(group[xaxis].unique()):
                if target != 'r_squared':
                    BH_res.append(group[group[xaxis] == x][f'BH_res_{target}'].values[0])
                    BH_rel.append(group[group[xaxis] == x][f'BH_rel_{target}'].values[0])
                    BH_2clip.append(group[group[xaxis] == x][f'BH_2clip_{target}'].values[0])
                    bon.append(group[group[xaxis] == x][f'Bonf_{target}'].values[0])
                else:
                    r_sq.append(group[group[xaxis] == x][f'r_squared'].values[0])
            x = idx // 4
            y = idx % 4
            if target != 'r_squared':
                if idx == 0:
                    # axs[x][y].plot(BH_res, marker='o', label="BH_res")
                    axs[x][y].plot(np.arange(*xrange), BH_rel, label="BH_sub")
                    axs[x][y].plot(np.arange(*xrange), BH_2clip, label="BH_2clip")
                    # axs[x][y].plot(bon, marker='o', label="Bonferroni")
                else:
                    # axs[x][y].plot(BH_res, marker='o')
                    axs[x][y].plot(np.arange(*xrange), BH_rel)
                    axs[x][y].plot(np.arange(*xrange), BH_2clip)
                    # axs[x][y].plot(bon, marker='o')
            else:
                if idx == 0:
                    axs[x][y].plot(np.arange(*xrange), r_sq, label="R^2")
                else:
                    axs[x][y].plot(np.arange(*xrange), r_sq)
        elif regressor in ['linear', 'additive']:
            # only plot horizontal lines
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
    if regressor in ['mlp', 'rf']:
        fig.supxlabel(f"{regressor} - {xaxis}")
    elif regressor in ['linear', 'additive']:
        fig.supxlabel(f"{regressor}")
    fig.supylabel(f'{tname}')
    fig.suptitle(f"{tname} for different procedures and settings with control level 0.1, noise level {sigma} with {regressor} regressor. \n {ntest} tests and {dim} total features, averged over {itr} times.")
    fig.legend()
    if regressor in ['mlp', 'rf']:
        plt.savefig(f'{regressor}-complexity {target} {xrange[0]},{xrange[1]},{xrange[2]} sigma={sigma} itr={itr} ntest={ntest} dim={dim}.png')
    elif regressor in ['linear', 'additive']:
        plt.savefig(f'{regressor} {target} interaction={interaction} sigma={sigma} itr={itr} ntest={ntest} dim={dim}.png')