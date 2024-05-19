"""
Created on Thu Mar 31 16:07:05 2022

@author: ying
"""

import numpy as np
import pandas as pd 
import random
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from utils import gen_data, BH

# hardcode the batchs
sig_list = [i / 10 for i in range(1, 11)]
nt_list = [10, 100, 500, 1000]
set_list = [1, 2, 3, 4, 5, 6, 7, 8]
seed_list = [i for i in range(0, 100)]
q_list = [0.1, 0.2, 0.5]

# # for testing:
# sig_list = [0.1]
# nt_list = [10]
# set_list = [1]
# seed_list = [0]
# q_list = [0.1]

reg_names = ['gbr', 'rf', 'svm']
    
n = 1000
all_res = pd.DataFrame()
out_dir = "../results/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print("Output diretory created!")

def run(all_res, sig, ntest, set, seed, q):
    random.seed(seed)

    for reg_name in reg_names:
        Xtrain, Ytrain, mu_train = gen_data(set, n, sig)
        Xcalib, Ycalib, mu_calib = gen_data(set, n, sig)
        
        Xtest, Ytest, mu_test = gen_data(set, ntest, sig)
        
        # training the prediction model
        if reg_name == 'gbr':
            regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
        elif reg_name == 'rf':
            regressor = RandomForestRegressor(max_depth=5, random_state=0)
        else:
            regressor = SVR(kernel="rbf", gamma=0.1)
        
        regressor.fit(Xtrain, 1 * (Ytrain > 0))
        
        # calibration 
        calib_scores = Ycalib - regressor.predict(Xcalib) 
        calib_scores0 = - regressor.predict(Xcalib) 
        calib_scores_clip = Ycalib * (Ycalib > 0) - regressor.predict(Xcalib)
        calib_scores_2clip = 1000 * (Ycalib > 0) - regressor.predict(Xcalib)
        
        test_scores = - regressor.predict(Xtest) 
        
        # BH using residuals
        BH_res= BH(calib_scores, test_scores, q )
        # summarize
        if len(BH_res) == 0:
            BH_res_fdp = 0
            BH_res_power = 0
        else:
            BH_res_fdp = np.sum(Ytest[BH_res] < 0) / len(BH_res)
            BH_res_power = np.sum(Ytest[BH_res] >= 0) / sum(Ytest >= 0)
            
        # only use relevant samples to calibrate
        BH_rel = BH(calib_scores0[Ycalib <= 0], test_scores, q )
        if len(BH_rel) == 0:
            BH_rel_fdp = 0
            BH_rel_power = 0
        else:
            BH_rel_fdp = np.sum(Ytest[BH_rel] < 0) / len(BH_rel)
            BH_rel_power = np.sum(Ytest[BH_rel] >= 0) / sum(Ytest >= 0)  
            
        # use clipped scores
        BH_2clip = BH(calib_scores_2clip, test_scores, q )
        if len(BH_2clip) == 0:
            BH_2clip_fdp = 0
            BH_2clip_power = 0
        else:
            BH_2clip_fdp = np.sum(Ytest[BH_2clip] < 0) / len(BH_2clip)
            BH_2clip_power = np.sum(Ytest[BH_2clip] >= 0) / sum(Ytest >= 0)
        
        all_res = pd.concat((all_res, 
                            pd.DataFrame({
                                        'sigma': [sig],
                                        'q': [q],
                                        'set': [set],
                                        'ntest': [ntest],
                                        'regressor': [reg_name],
                                        'seed': [seed],
                                        'BH_res_fdp': [BH_res_fdp], 
                                        'BH_res_power': [BH_res_power],
                                        'BH_res_nsel': [len(BH_res)],
                                        'BH_rel_fdp': [BH_rel_fdp], 
                                        'BH_rel_power': [BH_rel_power], 
                                        'BH_rel_nsel': [len(BH_rel)], 
                                        'BH_2clip_fdp': [BH_2clip_fdp], 
                                        'BH_2clip_power': [BH_2clip_power], 
                                        'BH_2clip_nsel': [len(BH_2clip)],
                                        })))
    return all_res

for sig in sig_list:
    for ntest in nt_list:
        for set in set_list:
            for seed in seed_list:
                for q in q_list:
                    all_res = run(all_res, sig, ntest, set, seed, q)

all_res.to_csv("../results/all.csv")
