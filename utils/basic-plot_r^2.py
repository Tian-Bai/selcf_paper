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

''' 
1. generate the FDR and power plot as in the paper
'''
regressor = 'gbr' # 'rf', 'gbr', 'svm'

target = "r_squared"
t2 = "R^2"

df = pd.read_csv(f"..\\csv\\{regressor}1256avgwithr^2.csv")

fig, axs = plt.subplots(figsize=(10, 10), nrows = 4, ncols = 4, sharex=True, sharey=True)

idx = 0
grouped = df.groupby(['set', 'ntest'])
# bonf_grouped = average_bonf_df.groupby(['set', 'ntest'])

for (s, n), group in grouped:
    r_sq = []
    for sigma in sorted(group['sigma'].unique()):
        r_sq.append(group[group['sigma'] == sigma]['r_squared'].values[0])

    x = idx // 4
    y = idx % 4
    axs[x][y].plot(r_sq, marker='o')
    
    if y % 4 == 3: # right
        axs[x][y].yaxis.set_label_position('right')
        #axs[x][y].yaxis.set_ticks_position('right')
        axs[x][y].set_ylabel(f'Setting {s if s < 5 else s - 2}')
    
    if x == 0: # top
        axs[x][y].xaxis.set_label_position('top')
        axs[x][y].set_xlabel(f'ntest = {n}')
    idx += 1

# fig.text(0.38, 0.06, f"{t2} for different procedures, number of tests and settings")
# fig.text(0.475, 0.08, "Noise level sigma")
fig.supxlabel("Noise level sigma")
fig.supylabel(f'{t2}')
fig.suptitle(f"{t2} for different procedures, number of tests and settings \n with control level 0.1, with {regressor} regressor")

plt.savefig(f'{target} {regressor}.png')