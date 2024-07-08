import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import random
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from utility import gen_data_2d, BH, Bonferroni
from prediction_model import OracleRegressor2d
import argparse
import itertools
from multiprocessing import Pool
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate data for oracle regressor experiment.')
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

# hardcode the sigma
sig_list = [0.5]
sig_list2 = [0.2]

covar_list = [0.1]
covar_list2 = [0.1]

set_list = [1, 2]
set_list2 = [5, 6]

seed_list = [i for i in range(0, itr)]

n = 1000 # train size
all_res = pd.DataFrame()
out_dir = f"..\\csv2d\\oracle"
full_out_dir = f"..\\csv2d\\oracle\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print("Output diretory created!")

''' 
distance from a point (y1, y2) to the set R^2/Q1.
'''
def dist(Y):
    l = []
    for i in range(len(Y)):
        if Y[i, 0] >= 0 and Y[i, 1] >= 0:
            l.append(max(Y[i, 0], Y[i, 1]))
        else:
            l.append(0)
    return np.array(l)

def run(sig, covar, setting, seed, **kwargs):
    df = pd.DataFrame()
    random.seed(seed)

    Xtrain, Ytrain, _, _, _ = gen_data_2d(setting, n, sig, covar, dim=dim)
    Xcalib, Ycalib, _, _, _ = gen_data_2d(setting, n, sig, covar, dim=dim)

    Xtest, Ytest, _, _, _ = gen_data_2d(setting, ntest, sig, covar, dim=dim)
    true_null = sum((Ytest[:, 0] > 0) & (Ytest[:, 1] > 0))
    # don't consider the no true null case (rejection sampling)
    # while true_null == 0:
    #     Xtest, Ytest, mu_test = gen_data(setting, ntest, sig)
    #     true_null = sum(Ytest > 0)

    # oracle regressor
    reg = OracleRegressor2d(setting)
    
    # fit (Y > 0) directly, not Y
    reg.fit(Xtrain, Ytrain)
    Ypred_calib = reg.predict(Xcalib)
    
    # calibration 
    calib_scores = dist(Ycalib) - dist(Ypred_calib)                                       # BH_res
    calib_scores0 = - dist(Ypred_calib)                                                   # BH_sub
    calib_scores_2clip = 1000 * dist(Ycalib) - dist(Ypred_calib)                          # BH_clip (with M = 1000)

    Ypred = reg.predict(Xtest) 
    test_scores = - dist(Ypred)

    r_sq = r2_score(Ytest, Ypred)

    # BH using residuals
    BH_res = BH(calib_scores, test_scores, q )
    # summarize
    if len(BH_res) == 0:
        BH_res_fdp = 0
        BH_res_power = 0
    else:
        BH_res_fdp = np.sum((Ytest[BH_res][:, 0] <= 0) | (Ytest[BH_res][:, 1] <= 0)) / len(BH_res)
        BH_res_power = np.sum((Ytest[BH_res][:, 0] > 0) & (Ytest[BH_res][:, 1] > 0)) / true_null if true_null != 0 else 0
        
    # only use relevant samples to calibrate
    BH_rel = BH(calib_scores0[(Ycalib[:, 0] <= 0) | (Ycalib[:, 1] <= 0)], test_scores, q )
    if len(BH_rel) == 0:
        BH_rel_fdp = 0
        BH_rel_power = 0
    else:
        BH_rel_fdp = np.sum((Ytest[BH_rel][:, 0] <= 0) | (Ytest[BH_rel][:, 1] <= 0)) / len(BH_rel)
        BH_rel_power = np.sum((Ytest[BH_rel][:, 0] > 0) & (Ytest[BH_rel][:, 1] > 0)) / true_null if true_null != 0 else 0
        
    # use clipped scores
    BH_2clip = BH(calib_scores_2clip, test_scores, q )
    if len(BH_2clip) == 0:
        BH_2clip_fdp = 0
        BH_2clip_power = 0
    else:
        BH_2clip_fdp = np.sum((Ytest[BH_2clip][:, 0] <= 0) | (Ytest[BH_2clip][:, 1] <= 0)) / len(BH_2clip)
        BH_2clip_power = np.sum((Ytest[BH_2clip][:, 0] > 0) & (Ytest[BH_2clip][:, 1] > 0)) / true_null if true_null != 0 else 0

    # Bonferroni
    Bonf = Bonferroni(calib_scores_2clip, test_scores, q )
    if len(Bonf) == 0:
        Bonf_fdp = 0
        Bonf_power = 0
    else:
        Bonf_fdp = np.sum((Ytest[Bonf][:, 0] <= 0) | (Ytest[Bonf][:, 1] <= 0)) / len(Bonf)
        Bonf_power = np.sum((Ytest[Bonf][:, 0] > 0) & (Ytest[Bonf][:, 1] > 0)) / true_null if true_null != 0 else 0

    df_dict = {
            'sigma': [sig],
            'dim': [dim],
            'q': [q],
            'set': [setting],
            'ntest': [ntest],
            'regressor': "oracle",
            'seed': [seed],
            'r_squared': [r_sq],
            'BH_res_fdp': [BH_res_fdp], 
            'BH_res_power': [BH_res_power],
            'BH_res_nsel': [len(BH_res)],
            'BH_rel_fdp': [BH_rel_fdp], 
            'BH_rel_power': [BH_rel_power], 
            'BH_rel_nsel': [len(BH_rel)], 
            'BH_2clip_fdp': [BH_2clip_fdp], 
            'BH_2clip_power': [BH_2clip_power], 
            'BH_2clip_nsel': [len(BH_2clip)],
            'Bonf_fdp': [Bonf_fdp], 
            'Bonf_power': [Bonf_power],
            'Bonf_nsel': [len(Bonf)],
            }
    df_dict.update(kwargs)
    
    df = pd.DataFrame(df_dict)
    return df

def run2(tup):
    sig, covar, setting, seed = tup
    return run(sig, covar, setting, seed)

if __name__ == '__main__':
    combined_itr = itertools.product(sig_list, covar_list, set_list, seed_list)
    combined_itr2 = itertools.product(sig_list2, covar_list2, set_list2, seed_list)
    total_len = len(sig_list) * len(covar_list) * len(set_list) * len(seed_list)
    total_len2 = len(sig_list2) * len(covar_list2) * len(set_list2) * len(seed_list)

    # multiprocessing version
    with Pool(processes=6) as pool:
        results = list(tqdm(pool.imap(run2, combined_itr), total=total_len))
    with Pool(processes=6) as pool:
        results2 = list(tqdm(pool.imap(run2, combined_itr2), total=total_len2))

    all_res = pd.concat(results, ignore_index=True)
    all_res2 = pd.concat(results2, ignore_index=True)
    all_res = pd.concat((all_res, all_res2), ignore_index=True)
                        
    all_res.to_csv(full_out_dir) 