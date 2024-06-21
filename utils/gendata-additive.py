import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import random
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from utils import gen_data, BH, Bonferroni, OracleRegressor
import argparse
import itertools
from multiprocessing import Pool
from tqdm import tqdm
from pygam import LinearGAM, s, te
from pygam.terms import TermList

parser = argparse.ArgumentParser(description='Generate data for GAM regressor experiment.')
parser.add_argument('-i', '--input', dest='itr', type=int, help='number of tests (seeds)', default=1000)
# parser.add_argument('-s', '--sigma', dest='sigma', type=str, help='sigma level', default='0.5(4)-0.2(4)')
parser.add_argument('-d', '--dim', dest='dim', type=int, help='number of features in generated data', default=20)
parser.add_argument('-n', '--ntest', dest='ntest', type=int, help='number of tests (m) in the setting', default=100)
parser.add_argument('--interaction', dest='interaction', type=bool, help='whether including interaction terms in the additive model', default=False)

args = parser.parse_args()

itr = args.itr
ntest = args.ntest
interaction = args.interaction
sigma = '0.5(4)-0.2(4)'
dim = args.dim
q = 0.1

# hardcode the sigma
sig_list = [0.5]
sig_list2 = [0.2]

set_list = [1, 2, 3, 4]
set_list2 = [5, 6, 7, 8]

seed_list = [i for i in range(0, itr)]

n = 1000 # train size
spec = "interaction" if interaction else "simple"
all_res = pd.DataFrame()
out_dir = f"..\\csv\\additive\\{spec}"
full_out_dir = f"..\\csv\\additive\\{spec}\\ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print("Output diretory created!")

def run(sig, setting, seed, **kwargs):
    df = pd.DataFrame()
    random.seed(seed)

    Xtrain, Ytrain, mu_train = gen_data(setting, n, sig, dim=dim)
    Xcalib, Ycalib, mu_calib = gen_data(setting, n, sig, dim=dim)

    Xtest, Ytest, mu_test = gen_data(setting, ntest, sig, dim=dim)
    true_null = sum(Ytest > 0)
    # don't consider the no true null case (rejection sampling)
    # while true_null == 0:
    #     Xtest, Ytest, mu_test = gen_data(setting, ntest, sig)
    #     true_null = sum(Ytest > 0)

    # GAM regressor
    if not interaction:
        # create a termlist and add all spline terms
        tm_list = TermList()
        for i in range(dim):
            tm_list += s(i)
        reg = LinearGAM(tm_list)
        # fit (Y > 0) directly, not Y
        reg.fit(Xtrain, 1 * (Ytrain > 0))
    else:
        # augment the data with interaction
        tm_list = TermList()
        for i in range(4): # alternatively, use 4 here
            tm_list += s(i)
        if setting == 1:
            tm_list += te(0, 1, 2)
        if setting in [2, 3, 4]:
            tm_list += te(0, 1)
        if setting == 5:
            tm_list += te(0, 1, 3)
        if setting in [6, 7, 8]:
            tm_list += te(0, 1)
        reg = LinearGAM(tm_list)
        # fit (Y > 0) directly, not Y
        reg.fit(Xtrain, 1 * (Ytrain > 0))    
        
    
    # calibration 
    calib_scores = Ycalib - reg.predict(Xcalib)                          # BH_res
    calib_scores0 = - reg.predict(Xcalib)                                # BH_sub
    calib_scores_clip = Ycalib * (Ycalib > 0) - reg.predict(Xcalib)
    calib_scores_2clip = 1000 * (Ycalib > 0) - reg.predict(Xcalib)       # BH_clip (with M = 1000)
    
    Ypred = reg.predict(Xtest) 
    test_scores = -Ypred

    r_sq = r2_score((Ytest > 0), Ypred)
    
    # BH using residuals
    BH_res = BH(calib_scores, test_scores, q )
    # summarize
    if len(BH_res) == 0:
        BH_res_fdp = 0
        BH_res_power = 0
    else:
        BH_res_fdp = np.sum(Ytest[BH_res] <= 0) / len(BH_res)
        BH_res_power = np.sum(Ytest[BH_res] > 0) / true_null if true_null != 0 else 0
        
    # only use relevant samples to calibrate
    BH_rel = BH(calib_scores0[Ycalib <= 0], test_scores, q )
    if len(BH_rel) == 0:
        BH_rel_fdp = 0
        BH_rel_power = 0
    else:
        BH_rel_fdp = np.sum(Ytest[BH_rel] <= 0) / len(BH_rel)
        BH_rel_power = np.sum(Ytest[BH_rel] > 0) / true_null if true_null != 0 else 0
        
    # use clipped scores
    BH_2clip = BH(calib_scores_2clip, test_scores, q )
    if len(BH_2clip) == 0:
        BH_2clip_fdp = 0
        BH_2clip_power = 0
    else:
        BH_2clip_fdp = np.sum(Ytest[BH_2clip] <= 0) / len(BH_2clip)
        BH_2clip_power = np.sum(Ytest[BH_2clip] > 0) / true_null if true_null != 0 else 0

    # Bonferroni
    Bonf = Bonferroni(calib_scores_2clip, test_scores, q )
    if len(Bonf) == 0:
        Bonf_fdp = 0
        Bonf_power = 0
    else:
        Bonf_fdp = np.sum(Ytest[Bonf] <= 0) / len(Bonf)
        Bonf_power = np.sum(Ytest[Bonf] > 0) / true_null if true_null != 0 else 0

    df_dict = {
            'sigma': [sig],
            'dim': [dim],
            'q': [q],
            'set': [setting],
            'ntest': [ntest],
            'regressor': f"additive-{spec}",
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
    sig, setting, seed = tup
    return run(sig, setting, seed)

multiproc = True

if __name__ == '__main__':
    combined_itr = itertools.product(sig_list, set_list, seed_list)
    combined_itr2 = itertools.product(sig_list2, set_list2, seed_list)
    total_len = len(sig_list) * len(set_list) * len(seed_list)
    total_len2 = len(sig_list2) * len(set_list2) * len(seed_list)

    if multiproc:
        # multiprocessing version
        with Pool(processes=6) as pool:
            results = list(tqdm(pool.imap(run2, combined_itr), total=total_len))
        with Pool(processes=6) as pool:
            results2 = list(tqdm(pool.imap(run2, combined_itr2), total=total_len2))

        all_res = pd.concat(results, ignore_index=True)
        all_res2 = pd.concat(results2, ignore_index=True)
        all_res = pd.concat((all_res, all_res2), ignore_index=True)
    else:
        # regular version (not updated)
        for (a, b, c, d, e, f, g, h) in tqdm(combined_itr, total=total_len):
            if c >= 5:
                a = 0.2
            df = run(a, b, c, d, e, f, g, h)
            all_res = pd.concat((all_res, df))
                        
    all_res.to_csv(full_out_dir) 