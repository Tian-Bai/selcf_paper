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

rf_param = ['n_estim', 'max_depth', 'max_features', 'max_leaf_nodes']
mlp_param = ['hidden', 'layers']

# parsers, and general configurations
parser = argparse.ArgumentParser(description='Plot 4 targets (FDP, power, nsel and r^2) for any specified regressor and test case.')
parser.add_argument('-i', '--input', dest='itr', type=int, help='number of tests (seeds)', default=1000)
# parser.add_argument('-s', '--sigma', dest='sigma', type=str, help='sigma level', default='0.5(4)-0.2(4)')
parser.add_argument('-d', '--dim', dest='dim', type=int, help='number of features in generated data', default=10)
parser.add_argument('-n', '--ntest', dest='ntest', type=int, help='number of tests (m) in the setting', default=100)
parser.add_argument('-c', '--continuous', dest='cont', type=str, help='whether consider the data as continuous or not', default='False')

# subparsers for different supported models
subparsers = parser.add_subparsers(dest='regressor', required=True, help='The target regressor. Choose between ["rf", "mlp", "additive", "linear", ...].')
parser_rf = subparsers.add_parser('rf', help='rf regressor parser.')
parser_mlp = subparsers.add_parser('mlp', help='mlp regressor parser.')
parser_linear = subparsers.add_parser('linear', help='linear regressor parser.')
parser_additive = subparsers.add_parser('additive', help='GAM regressor parser.')
parser_oracle = subparsers.add_parser('oracle', help='Oracle regressor parser.')

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

cont = args.cont
regressor = args.regressor
itr = args.itr
ntest = args.ntest
sigma = '0.5(4)-0.2(4)'
dim = args.dim
q = 0.1

targets = [('fdp', 'FDP'), ('power', 'Power'), ('nsel', 'Number of rejections'), ('r_squared', 'Out of sample R^2'), ('accuracy', 'Accuracy')]

if regressor == 'rf':
    xaxis = args.xaxis
    xrange = args.range
    config = args.config

    rf_param2 = [r for r in rf_param]
    rf_param2.remove(xaxis)
    df = pd.read_csv(f"..\\csv\\cont={cont}\\{regressor}\\{xaxis}\\{xrange[0]},{xrange[1]},{xrange[2]} {rf_param2[0]}={config[rf_param2[0]]} {rf_param2[1]}={config[rf_param2[1]]} {rf_param2[2]}={config[rf_param2[2]]} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv", keep_default_na=False, na_values=[])
    gb = ['set', 'regressor', 'dim'] + rf_param
elif regressor == 'mlp':
    xaxis = args.xaxis
    xrange = args.range
    config = args.config

    mlp_param2 = [r for r in mlp_param]
    mlp_param2.remove(xaxis)
    df = pd.read_csv(f"..\\csv\\cont={cont}\\{regressor}\\{xaxis}\\{xrange[0]},{xrange[1]},{xrange[2]} {mlp_param2[0]}={config[mlp_param2[0]]} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
    gb = ['set', 'regressor', 'dim'] + mlp_param

    # whether to use number of parameters as the xaxis?
    # input: d
    # output: 1
    # (d+1) * hidden + (layers-1) * (hidden+1) * (hidden) + (hidden+1) * 1
    # if no hidden layer:
    # (d+1) * 1
    xlist = np.arange(*xrange)
    plotxlist = []
    if xaxis == 'hidden': # at least 1 hidden layers
        layers = int(config['layers'])
        for x in xlist:
            if layers != 0:
                plotxlist.append((dim + 1) * x + (layers - 1) * (x + 1) * x + (x + 1) * 1)
            else:
                plotxlist.append((dim + 1) * 1)
    elif xaxis == 'layers':
        hidden = int(config['hidden'])
        for x in xlist:
            if x != 0:
                plotxlist.append((dim + 1) * hidden + (x - 1) * (hidden + 1) * hidden + (hidden + 1) * 1)
            else:
                plotxlist.append((dim + 1) * 1)
elif regressor in ['linear', 'additive']:
    interaction = args.interaction

    df = pd.read_csv(f"..\\csv\\cont={cont}\\{regressor}\\interaction={interaction}\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
    gb = ['set', 'regressor', 'dim', 'interaction']
elif regressor == 'oracle':
    df = pd.read_csv(f"..\\csv2d\\cont={cont}\\{regressor}\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
    gb = ['set', 'regressor', 'dim']
df = df.groupby(gb).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

grouped = df.groupby(['set'])

plotxaxis = np.arange(*xrange)
# for mlp, could choose to use model parameters as xaxis
if regressor == 'mlp':
    plotxaxis = plotxlist

only1256 = True

for (target, tname) in targets:
    fig, axs = plt.subplots(figsize=(20, 10), nrows = 2, ncols = 2 if only1256 else 4, sharex=True, sharey=True)
    idx = 0

    for (s,), group in grouped:
        if only1256 and s not in [1, 2, 5, 6]:
            continue
        if regressor in ['rf', 'mlp'] and len(group[xaxis].unique()) > 1:
            BH_res = []
            BH_rel = []
            BH_2clip = []
            bon = []
            r_sq = []
            accuracy = []
            for x in sorted(group[xaxis].unique()):
                if target not in ['r_squared', 'accuracy']:
                    BH_res.append(group[group[xaxis] == x][f'BH_res_{target}'].values[0])
                    BH_rel.append(group[group[xaxis] == x][f'BH_rel_{target}'].values[0])
                    BH_2clip.append(group[group[xaxis] == x][f'BH_2clip_{target}'].values[0])
                    bon.append(group[group[xaxis] == x][f'Bonf_{target}'].values[0])
                elif target == 'r_squared':
                    r_sq.append(group[group[xaxis] == x][f'r_squared'].values[0])
                else:
                    accuracy.append(group[group[xaxis] == x][f'accuracy'].values[0])
            x = idx // (2 if only1256 else 4)
            y = idx % (2 if only1256 else 4)
            if target not in ['r_squared', 'accuracy']:
                if idx == 0:
                    # axs[x][y].plot(BH_res, marker='o', label="BH_res")
                    axs[x][y].plot(plotxaxis, BH_rel, label="BH_sub")
                    axs[x][y].plot(plotxaxis, BH_2clip, label="BH_2clip")
                    # axs[x][y].plot(bon, marker='o', label="Bonferroni")
                else:
                    # axs[x][y].plot(BH_res, marker='o')
                    axs[x][y].plot(plotxaxis, BH_rel)
                    axs[x][y].plot(plotxaxis, BH_2clip)
                    # axs[x][y].plot(bon, marker='o')
            elif target == 'r_squared':
                if idx == 0:
                    axs[x][y].plot(plotxaxis, r_sq, label="R^2")
                else:
                    axs[x][y].plot(plotxaxis, r_sq)
            else:
                if idx == 0:
                    axs[x][y].plot(plotxaxis, accuracy, label="Accuracy")
                else:
                    axs[x][y].plot(plotxaxis, accuracy)
            # axs[x][y].set_xticks(plotxaxis, [(str(j) if i % 2 == 0 else "") for i, j in enumerate(plotxaxis)], fontsize=7)
            #
        else:
            # only plot horizontal lines
            if target not in ['r_squared', 'accuracy']:
                BH_res = group[f'BH_res_{target}'].values[0]
                BH_rel = group[f'BH_rel_{target}'].values[0]
                BH_2clip = group[f'BH_2clip_{target}'].values[0]
                bon = group[f'Bonf_{target}'].values[0]
            r_sq = group[f'r_squared'].values[0]
            accuracy = group[f'accuracy'].values[0]
            x = idx // 4
            y = idx % 4
            if target not in ['r_squared', 'accuracy']:
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
            elif target == 'r_squared':
                axs[x][y].axhline(y=r_sq)
            else:
                axs[x][y].axhline(y=accuracy)
        axs[x][y].set_xlabel(f'Setting {s if (only1256 and s > 4) else s}')
        if target == 'power':
            axs[x][y].set_ylim((0, 1.1))
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
    elif regressor == 'oracle':
        plt.savefig(f'{regressor} {target} sigma={sigma} itr={itr} ntest={ntest} dim={dim}.png')