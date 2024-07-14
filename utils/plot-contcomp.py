import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import argparse

# compare many similar cases (oracle).

parser = argparse.ArgumentParser(description='Experiment specifications.')
parser.add_argument('-i', '--input', dest='itr', type=int, help='number of tests (seeds)', default=1000)
# parser.add_argument('-s', '--sigma', dest='sigma', type=str, help='sigma level', default='0.5(4)-0.2(4)')
parser.add_argument('-d', '--dim', dest='dim', type=int, help='number of features in generated data', default=10)
parser.add_argument('-n', '--ntest', dest='ntest', type=int, help='number of tests (m) in the setting', default=100)

args = parser.parse_args()

itr = args.itr
ntest = args.ntest
sigma = '0.1'
cov = '0.1'
dim = args.dim
q = 0.1

targets = [('fdp', 'FDP'), ('power', 'Power'), ('nsel', 'Number of rejections'), ('r_squared', 'Out of sample R^2')] # 'power', 'nsel'

oracle_1d_c_df = pd.read_csv(f"..\\csv\\cont=True\\oracle\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
oracle_1d_c_df = oracle_1d_c_df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

oracle_1d_d_df = pd.read_csv(f"..\\csv\\cont=False\\oracle\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv")
oracle_1d_d_df = oracle_1d_d_df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

oracle_2d_c_df = pd.read_csv(f"..\\csv2d\\cont=True\\oracle\\ntest={ntest} itr={itr} sigma={sigma} cov={cov} dim={dim}.csv")
oracle_2d_c_df = oracle_2d_c_df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

oracle_2d_d_df = pd.read_csv(f"..\\csv2d\\cont=False\\oracle\\ntest={ntest} itr={itr} sigma={sigma} cov={cov} dim={dim}.csv")
oracle_2d_d_df = oracle_2d_d_df.groupby(['set', 'regressor', 'dim']).mean().reset_index().drop(columns=['Unnamed: 0', 'seed'])

combined_df = pd.concat([oracle_1d_c_df, oracle_1d_d_df, oracle_2d_c_df, oracle_2d_d_df], axis=0, ignore_index=True)

experiment_name = ['1d-cont.', '1d-disc.', '2d-s.c.', '2d-s.d.', '2d-d.c.', '2d-d.d.']
# setting: 1, 1, (1 & 5), (1 & 5) / 2, 2, (2 & 6), (2 & 6) / 5, 5, (3 & 7), (3 & 7) / 6, 6, (4 & 8), (4 & 8)
setting_iloc = [[0, 8, 16, 20, 24, 28], [1, 9, 17, 21, 25, 29], [4, 12, 18, 22, 26, 30], [5, 13, 19, 23, 27, 31]]

for (target, tname) in targets:
    fig, axs = plt.subplots(figsize=(14, 10), nrows = 2, ncols = 2, sharex=True, sharey=True)
    idx = 0

    for i, settings in enumerate(setting_iloc):
        BH_rel = []
        BH_2clip = []
        r_sq = []
        if target != 'r_squared':
            for s in settings:
                BH_rel.append(combined_df.iloc[s][f'BH_rel_{target}'])
                BH_2clip.append(combined_df.iloc[s][f'BH_2clip_{target}'])
        else:
            for s in settings:
                r_sq.append(combined_df.iloc[s][f'r_squared'])
                
        x = idx // 2
        y = idx % 2
        if target != 'r_squared':
            if idx == 0:
                axs[x][y].bar(np.arange(len(experiment_name)), BH_rel, 0.2, label='BH_sub')
                axs[x][y].bar(np.arange(len(experiment_name)) + 0.2, BH_2clip, 0.2, label='BH_2clip')
            else:
                axs[x][y].bar(np.arange(len(experiment_name)), BH_rel, 0.2)
                axs[x][y].bar(np.arange(len(experiment_name)) + 0.2, BH_2clip, 0.2)
        else:
            if idx == 0:
                axs[x][y].bar(np.arange(len(experiment_name)) + 0.1, r_sq, 0.3, label='BH_sub')
            else:
                axs[x][y].bar(np.arange(len(experiment_name)) + 0.1, r_sq, 0.3)
        axs[x][y].set_xticks(np.arange(len(experiment_name)) + 0.1, experiment_name)
        axs[x][y].set_xlabel(f'Setting {i+1}')
        
        idx += 1

    fig.supxlabel("For comparison")
    fig.supylabel(f'{tname}')
    fig.suptitle(f"{tname} for different experiment settings with oracle regressor, control level 0.1, noise level {sigma}. \n {ntest} tests and {dim} total features, averaged over {itr} times.")
    fig.legend()
    plt.savefig(f'contcomp {target} sigma={sigma} itr={itr} ntest={ntest} dim={dim}.png')