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
from sklearn.linear_model import QuantileRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from utility import gen_data, BH, Bonferroni
from utility import rf_config, rf_str, mlp_config, mlp_str, interaction_type, range_arg
import argparse
import itertools
from multiprocessing import Pool
from tqdm import tqdm
from pygam import LinearGAM, s, te
from pygam.terms import TermList
from quantile_forest import RandomForestQuantileRegressor

rf_param = ['n_estim', 'max_depth', 'max_features']
mlp_param = ['hidden', 'layers']

''' 
If the regressor is rf, parameters are ['n_estim', 'max_depth', 'max_features'].
If the regressor if mlp, parameters are ['hidden', 'layers'].
'''

# parsers, and general configurations
parser = argparse.ArgumentParser(description='Generate data for 4 targets (FDP, power, nsel and r^2) for any specified regressor and test case.')
parser.add_argument('-i', '--input', dest='itr', type=int, help='number of tests (seeds)', default=1000)
# parser.add_argument('-s', '--sigma', dest='sigma', type=str, help='sigma level', default='0.5(4)-0.2(4)')
parser.add_argument('-d', '--dim', dest='dim', type=int, help='number of features in generated data', default=10)
parser.add_argument('-n', '--ntest', dest='ntest', type=int, help='number of tests (m) in the setting', default=100)
parser.add_argument('-qt', '--quantile', dest='qt', type=range_arg, help='quantile range to consider', required=True)

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

# hardcode the sigma
sig_list = [0.5]
sig_list2 = [0.2]

set_list = [1, 2, 3, 4]
set_list2 = [5, 6, 7, 8]

# # trivial setting
# set_list = [0]

seed_list = [i for i in range(0, itr)]

n = 1000 # train size
all_res = pd.DataFrame()

if regressor == 'rf':
    config = args.config

    out_dir = f"..\\csv\\quantile-{regressor}\\"
    full_out_dir = f"..\\csv\\quantile-{regressor}\\{qt_list[0]},{qt_list[1]},{qt_list[2]} {rf_param[0]}={config[rf_param[0]]} {rf_param[1]}={config[rf_param[1]]} {rf_param[2]}={config[rf_param[2]]} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv"
elif regressor == 'mlp':
    config = args.config

    out_dir = f"..\\csv\\quantile-{regressor}\\"
    full_out_dir = f"..\\csv\\quantile-{regressor}\\{qt_list[0]},{qt_list[1]},{qt_list[2]} {mlp_param[0]}={config[mlp_param[0]]} {mlp_param[1]}={config[mlp_param[1]]} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv"
elif regressor in ['linear', 'additive']:
    interaction = args.interaction

    out_dir = f"..\\csv\\quantile-{regressor}\\"
    full_out_dir = f"..\\csv\\quantile-{regressor}\\{qt_list[0]},{qt_list[1]},{qt_list[2]} interaction={interaction} ntest={ntest} itr={itr} sigma={sigma} dim={dim}.csv"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print("Output diretory created!")

def run(sig, setting, seed, qt):            
    df = pd.DataFrame()
    random.seed(seed)

    Xtrain, Ytrain, mu_train = gen_data(setting, n, sig, dim=dim)
    Xcalib, Ycalib, mu_calib = gen_data(setting, n, sig, dim=dim)
    Xtest, Ytest, mu_test = gen_data(setting, ntest, sig, dim=dim)
    true_null = sum(Ytest > 0)
    
    # MLP with 'layer' hidden layer, each of size 'hidden'
    if regressor == 'rf':
        n_estim = int(config["n_estim"])
        max_depth = int(config["max_depth"])
        max_features = config["max_features"]
        try:
            max_features = int(max_features)
        except ValueError:
            pass # input is 'sqrt' or 'log2'.
        ''' 
        TODO: refactor this part to use the wrapper predictor
        '''
        if setting == 9:
            reg = RandomForestQuantileRegressor(default_quantiles=qt, min_samples_leaf=30, n_estimators=n_estim, max_depth=max_depth, max_features=max_features, random_state=0)
        else:
            reg = RandomForestQuantileRegressor(default_quantiles=qt, n_estimators=n_estim, max_depth=max_depth, max_features=max_features, random_state=0)
    elif regressor == 'mlp':
        raise NotImplementedError("To be implemented.")
    
        hidden = int(config["hidden"])
        layers = int(config["layers"])
        # TODO: implement MLPQuantileRegressor
        reg = MLPRegressor(hidden_layer_sizes=(hidden, ) * layers, random_state=0, alpha=3e-2, max_iter=1000)
    elif regressor == 'linear':
        if interaction == "no":
            # no interaction
            reg = QuantileRegressor(quantile=qt, alpha=0, solver='highs')
        elif interaction == "yes":
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            Xtrain = poly.fit_transform(Xtrain)
            Xcalib = poly.fit_transform(Xcalib)
            Xtest = poly.fit_transform(Xtest)
            reg = QuantileRegressor(quantile=qt, alpha=0, solver='highs')
        elif interaction == "oracle":
            if setting == 1:
                transf = lambda x : np.column_stack((x, x[:, 0] * x[:, 1], x[:, 0] * x[:, 2], x[:, 1] * x[:, 2], x[:, 0] * x[:, 1] * x[:, 2]))
            if setting in [2, 3, 4]:
                transf = lambda x : np.column_stack((x, x[:, 0] * x[:, 1]))
            if setting == 5:
                transf = lambda x : np.column_stack((x, x[:, 0] * x[:, 1], x[:, 0] * x[:, 3], x[:, 1] * x[:, 3], x[:, 0] * x[:, 1] * x[:, 3]))
            if setting in [6, 7, 8]:
                transf = lambda x : np.column_stack((x, x[:, 0] * x[:, 1]))
            Xtrain = transf(Xtrain)
            Xcalib = transf(Xcalib)
            Xtest = transf(Xtest)
            reg = QuantileRegressor(quantile=qt, alpha=0, solver='highs')
    elif regressor == 'additive':
        raise NotImplementedError("To be implemented. A relevant package in R is https://github.com/mfasiolo/qgam.")

        if interaction == "no":
            tm_list = TermList()
            for i in range(dim):
                tm_list += s(i)
            reg = LinearGAM(tm_list)
        elif interaction == "yes":
            tm_list = TermList()
            for i in range(dim):
                tm_list += s(i)
                for j in range(i+1, dim):
                    tm_list += te(i, j)
            reg = LinearGAM(tm_list)
        elif interaction == "oracle":
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
    
    if 1 <= setting and setting <= 8:
        # fit (Y > 0) directly, not Y
        reg.fit(Xtrain, 1 * (Ytrain > 0))
        
        # calibration 
        calib_scores = Ycalib - reg.predict(Xcalib)                          # BH_res
        calib_scores0 = - reg.predict(Xcalib)                                # BH_sub
        calib_scores_2clip = 1000 * (Ycalib > 0) - reg.predict(Xcalib)       # BH_clip (with M = 1000)
        
        Ypred = reg.predict(Xtest) 
        test_scores = -Ypred

        r_sq = r2_score((Ytest > 0), Ypred)
    else:
        reg.fit(Xtrain, Ytrain)
        
        # calibration 
        calib_scores = Ycalib - reg.predict(Xcalib)                          # BH_res
        calib_scores0 = - reg.predict(Xcalib)                                # BH_sub
        calib_scores_2clip = 1000 * (Ycalib > 0) - reg.predict(Xcalib)       # BH_clip (with M = 1000)
        
        Ypred = reg.predict(Xtest) 
        test_scores = -Ypred

        r_sq = r2_score(Ytest, Ypred)
    
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
            'regressor': [f"quantile-{regressor}"],
            'qt': [qt],
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
    
    df = pd.DataFrame(df_dict)
    return df

def run2(tup):
    sig, setting, seed, qt = tup
    return run(sig, setting, seed, qt)

if __name__ == '__main__':
    combined_itr = itertools.product(sig_list, set_list, seed_list, np.linspace(*qt_list))
    combined_itr2 = itertools.product(sig_list2, set_list2, seed_list, np.linspace(*qt_list))
    total_len = len(sig_list) * len(set_list) * len(seed_list) * len(np.linspace(*qt_list))
    total_len2 = len(sig_list2) * len(set_list2) * len(seed_list) * len(np.linspace(*qt_list))

    with Pool(processes=6) as pool:
        results = list(tqdm(pool.imap(run2, combined_itr), total=total_len))
    # with Pool(processes=6) as pool:
    #     results2 = list(tqdm(pool.imap(run2, combined_itr2), total=total_len2))

    all_res = pd.concat(results, ignore_index=True)
    # all_res2 = pd.concat(results2, ignore_index=True)
    # all_res = pd.concat((all_res, all_res2), ignore_index=True)
                        
    all_res.to_csv(full_out_dir) 