from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, accuracy_score
from utility import BH, Bonferroni, SingleSel
import random
import itertools
from multiprocessing import Pool
from tqdm import tqdm

def run(seed, q):
    train = pd.read_csv("data\\PGP_training_disguised.csv")
    n = len(train)

    random.seed(seed)
    perm = np.random.permutation(n)
    test = train.iloc[perm[:int(0.1 * n)]]
    # alternatively,
    # test = pd.read_csv("data\\PGP_test_disguised.csv")
    # test = test.reindex(columns=train.columns)
    
    calib = train.iloc[perm[int(0.1 * n):int(0.3 * n)]]
    train = train.iloc[perm[int(0.3 * n):]]

    Ytrain, Ycalib, Ytest = train['Act'].to_numpy(), calib['Act'].to_numpy(), test['Act'].to_numpy()
    Xtrain, Xcalib, Xtest = train.drop(columns=['MOLECULE', 'Act']).to_numpy(), calib.drop(columns=['MOLECULE', 'Act']).to_numpy(), test.drop(columns=['MOLECULE', 'Act']).to_numpy()

    lower = -0.3
    higher = 0.2

    Ytrain = 1 * ((lower < Ytrain) & (Ytrain < higher))
    Ycalib = 1 * ((lower < Ycalib) & (Ycalib < higher))
    Ytest = 1 * ((lower < Ytest) & (Ytest < higher))

    rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    rf.fit(Xtrain, Ytrain)

    r_sq = r2_score(Ytest, rf.predict(Xtest))

    calib_scores = Ycalib - rf.predict(Xcalib)          
    calib_scores0 = -rf.predict(Xcalib)                     
    calib_scores_2clip = 1000 * (Ycalib > 0) - rf.predict(Xcalib)   # Ycalib > 0.5 <=> original Ycalib < threshold
    
    test_scores = -rf.predict(Xtest)

    # selection target is 
    true_reject = sum(Ytest > 0) 

    # BH using residuals
    BH_res = BH(calib_scores, test_scores, q )
    # summarize
    if len(BH_res) == 0:
        BH_res_fdp = 0
        BH_res_power = 0
        BH_res_wrong = 0
    else:
        BH_res_fdp = np.sum(Ytest[BH_res] <= 0) / len(BH_res)
        BH_res_power = np.sum(Ytest[BH_res] > 0) / true_reject if true_reject != 0 else 0
        BH_res_wrong = np.sum(Ytest[BH_res] <= 0)
        
    # only use relevant samples to calibrate
    BH_rel = BH(calib_scores0[Ycalib <= 0], test_scores, q )
    if len(BH_rel) == 0:
        BH_rel_fdp = 0
        BH_rel_power = 0
        BH_rel_wrong = 0
    else:
        BH_rel_fdp = np.sum(Ytest[BH_rel] <= 0) / len(BH_rel)
        BH_rel_power = np.sum(Ytest[BH_rel] > 0) / true_reject if true_reject != 0 else 0
        BH_rel_wrong = np.sum(Ytest[BH_rel] <= 0)
        
    # use clipped scores
    BH_2clip = BH(calib_scores_2clip, test_scores, q )
    if len(BH_2clip) == 0:
        BH_2clip_fdp = 0
        BH_2clip_power = 0
        BH_2clip_wrong = 0
    else:
        BH_2clip_fdp = np.sum(Ytest[BH_2clip] <= 0) / len(BH_2clip)
        BH_2clip_power = np.sum(Ytest[BH_2clip] > 0) / true_reject if true_reject != 0 else 0
        BH_2clip_wrong = np.sum(Ytest[BH_2clip] <= 0)

    # Bonferroni
    Bonf = Bonferroni(calib_scores_2clip, test_scores, q )
    if len(Bonf) == 0:
        Bonf_fdp = 0
        Bonf_power = 0
        Bonf_wrong = 0
    else:
        Bonf_fdp = np.sum(Ytest[Bonf] <= 0) / len(Bonf)
        Bonf_power = np.sum(Ytest[Bonf] > 0) / true_reject if true_reject != 0 else 0
        Bonf_wrong = np.sum(Ytest[Bonf] <= 0)

    Single = SingleSel(calib_scores_2clip, test_scores, q )
    if len(Single) == 0:
        Single_fdp = 0
        Single_power = 0
        Single_wrong = 0
    else:
        Single_fdp = np.sum(Ytest[Single] <= 0) / len(Single)
        Single_power = np.sum(Ytest[Single] > 0) / true_reject if true_reject != 0 else 0
        Single_wrong = np.sum(Ytest[Single] <= 0)

    df_dict = {
            'q': [q],
            'regressor': ['rf'],
            'r_squared': [r_sq],
            'BH_res_fdp': [BH_res_fdp], 
            'BH_res_power': [BH_res_power],
            'BH_res_nsel': [len(BH_res)],
            'BH_res_wrong': [BH_rel_wrong],
            'BH_rel_fdp': [BH_rel_fdp], 
            'BH_rel_power': [BH_rel_power], 
            'BH_rel_nsel': [len(BH_rel)], 
            'BH_rel_wrong': [BH_rel_wrong],
            'BH_2clip_fdp': [BH_2clip_fdp], 
            'BH_2clip_power': [BH_2clip_power], 
            'BH_2clip_nsel': [len(BH_2clip)],
            'BH_2clip_wrong': [BH_2clip_wrong],
            'Bonf_fdp': [Bonf_fdp], 
            'Bonf_power': [Bonf_power],
            'Bonf_nsel': [len(Bonf)],
            'Bonf_wrong': [Bonf_wrong],
            'Single_fdp': [Single_fdp],
            'Single_power': [Single_power],
            'Single_nsel': [len(Single)],
            'Single_wrong': [Single_wrong]
            }
    df = pd.DataFrame(df_dict)
    return df

def run2(tup):
    seed, q = tup
    return run(seed, q)

seed_list = [i for i in range(100)]
q_list = [0.1, 0.2, 0.5]

if __name__ == '__main__':
    combined_itr = itertools.product(seed_list, q_list)
    total_len = len(seed_list) * len(q_list)

    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(run2, combined_itr), total=total_len))

    all_res = pd.concat(results, ignore_index=True)                   
    all_res.to_csv("PGP_result.csv") 