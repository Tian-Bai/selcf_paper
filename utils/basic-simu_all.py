import numpy as np
import pandas as pd 
import random
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from utils import gen_data, BH, Bonferroni
import itertools
from tqdm import tqdm

# hardcode the batchs
sig_list = [i / 10 for i in range(1, 11)]
nt_list = [10, 100, 500, 1000]
set_list = [1, 2, 5, 6]
seed_list = [i for i in range(0, 100)]
q_list = [0.1, 0.2, 0.5]

# # for testing:
# sig_list = [0.1]
# nt_list = [10]
# set_list = [1]
# seed_list = [0]
# q_list = [0.1]

reg_names = ['gbr', 'rf', 'svm']
reg_names = ['gbr']
    
n = 1000
all_res = pd.DataFrame()
out_dir = "../csv/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print("Output diretory created!")

def run(all_res, sig, ntest, set, seed, q):
    random.seed(seed)

    for reg_name in reg_names:
        Xtrain, Ytrain, mu_train = gen_data(set, n, sig)
        Xcalib, Ycalib, mu_calib = gen_data(set, n, sig)

        # don't consider the no true null case (rejection sampling)
        true_null = 0
        while true_null == 0:
            Xtest, Ytest, mu_test = gen_data(set, ntest, sig)
            true_null = sum(Ytest > 0)
        
        # training the prediction model
        if reg_name == 'gbr':
            regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
        elif reg_name == 'rf':
            regressor = RandomForestRegressor(max_depth=5, random_state=0)
        else:
            regressor = SVR(kernel="rbf", gamma=0.1)
        
        # fit (Y > 0) directly, not Y
        regressor.fit(Xtrain, 1 * (Ytrain > 0))
        
        # calibration 
        calib_scores = Ycalib - regressor.predict(Xcalib)                          # BH_res
        calib_scores0 = - regressor.predict(Xcalib)                                # BH_sub
        calib_scores_clip = Ycalib * (Ycalib > 0) - regressor.predict(Xcalib)
        calib_scores_2clip = 1000 * (Ycalib > 0) - regressor.predict(Xcalib)       # BH_clip (with M = 1000)
        
        Ypred = regressor.predict(Xtest) 
        test_scores = -Ypred

        r_sq = r2_score((Ytest > 0), Ypred)
        
        # BH using residuals
        BH_res= BH(calib_scores, test_scores, q )
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
        
        all_res = pd.concat((all_res, 
                            pd.DataFrame({
                                        'sigma': [sig],
                                        'q': [q],
                                        'set': [set],
                                        'ntest': [ntest],
                                        'regressor': [reg_name],
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
                                        })))
    return all_res

combined_itr = itertools.product(sig_list, nt_list, set_list, seed_list, q_list)
total_len = len(sig_list) * len(nt_list) * len(set_list) * len(seed_list) * len(q_list)

for (a, b, c, d, e) in tqdm(combined_itr, total=total_len):
# for (a, b, c, d, e) in combined_itr:
    all_res = run(all_res, a, b, c, d, e)
                    
all_res.to_csv("../csv/all.csv")
