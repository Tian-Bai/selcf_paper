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
from sklearn.metrics import r2_score, accuracy_score
from utility import gen_data, gen_data_2d, BH, Bonferroni
from utility import rf_config, rf_str, mlp_config, mlp_str, interaction_type, range_arg
from prediction_model import OracleRegressor2d_singledim, RfRegressor, MlpRegressor
import argparse
import itertools
from multiprocessing import Pool
from tqdm import tqdm
from pygam import LinearGAM, s, te
from pygam.terms import TermList

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
sigma = '0.1'
cov = '0.1'
dim = args.dim
q = 0.1 / 2

# hardcode the sigma and covar
sig_list = [0.1]
sig_list2 = [0.1]

covar_list = [0.1]
covar_list2 = [0.1]

set_list = [1, 2, 3, 4]
set_list2 = [5, 6, 7, 8]

seed_list = [i for i in range(0, itr)]

n = 1000 # train size
all_res = pd.DataFrame()

if regressor == 'rf':
    xaxis = args.xaxis
    xrange = args.range
    config = args.config

    rf_param2 = [r for r in rf_param]
    rf_param2.remove(xaxis)
    out_dir = f"..\\csv2d-naive\\cont={cont}\\{regressor}\\{xaxis}"
    full_out_dir = f"..\\csv2d-naive\\cont={cont}\\{regressor}\\{xaxis}\\{xrange[0]},{xrange[1]},{xrange[2]} {rf_param2[0]}={config[rf_param2[0]]} {rf_param2[1]}={config[rf_param2[1]]} ntest={ntest} itr={itr} sigma={sigma} cov={cov} dim={dim}.csv"
elif regressor == 'mlp':
    xaxis = args.xaxis
    xrange = args.range
    config = args.config

    mlp_param2 = [r for r in mlp_param]
    mlp_param2.remove(xaxis)
    out_dir = f"..\\csv2d-naive\\cont={cont}\\{regressor}\\{xaxis}"
    full_out_dir = f"..\\csv2d-naive\\cont={cont}\\{regressor}\\{xaxis}\\{xrange[0]},{xrange[1]},{xrange[2]} {mlp_param2[0]}={config[mlp_param2[0]]} ntest={ntest} itr={itr} sigma={sigma} cov={cov} dim={dim}.csv"
elif regressor in ['linear', 'additive']:
    interaction = args.interaction

    out_dir = f"..\\csv2d-naive\\cont={cont}\\{regressor}\\interaction={interaction}"
    full_out_dir = f"..\\csv2d-naive\\cont={cont}\\{regressor}\\interaction={interaction}\\ntest={ntest} itr={itr} sigma={sigma} cov={cov} dim={dim}.csv"
elif regressor == 'oracle':
    out_dir = f"..\\csv2d-naive\\cont={cont}\\{regressor}"
    full_out_dir = f"..\\csv2d-naive\\cont={cont}\\{regressor}\\ntest={ntest} itr={itr} sigma={sigma} cov={cov} dim={dim}.csv" 

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
        elif Y[i, 0] < 0 and Y[i, 1] < 0:
            l.append(-np.sqrt(Y[i, 0] ** 2 + Y[i, 1] ** 2))
        else:
            l.append(min(Y[i, 0], Y[i, 1]))
    return np.array(l)

def run(sig, covar, setting, seed, **kwargs):
    if regressor == 'rf':
        assert set(kwargs.keys()) <= set(rf_param)
    elif regressor == 'mlp':
        assert set(kwargs.keys()) <= set(mlp_param)
    elif regressor in ['linear', 'additive']:
        assert set(kwargs.keys()) <= set(['interaction'])
            
    df = pd.DataFrame()
    random.seed(seed)

    Xtrain, Ytrain, mu1_train, mu2_train, cov_train = gen_data_2d(setting, n, sig, covar, dim=dim)
    Xcalib, Ycalib, mu1_calib, mu2_calib, cov_calib = gen_data_2d(setting, n, sig, covar, dim=dim)

    Xtest, Ytest, mu1_test, mu2_test, cov_test = gen_data_2d(setting, ntest, sig, covar, dim=dim)

    # suppose the null hypothesis is Y1 or Y2 <= 0
    true_null = sum((Ytest[:, 0] > 0) & (Ytest[:, 1] > 0))
    
    # for the first dimension:
    if regressor == 'rf':
        n_estim = int(kwargs["n_estim"])
        max_depth = int(kwargs["max_depth"])
        max_features = kwargs["max_features"]
        max_leaf_nodes = kwargs["max_leaf_nodes"]
        try:
            max_features = int(max_features)
        except ValueError:
            pass # input is 'sqrt' or 'log2'.
        try:
            max_leaf_nodes = int(max_leaf_nodes)
        except ValueError:
            max_leaf_nodes = None # no restriction

        reg = RfRegressor(setting=setting, n_estimators=n_estim, max_depth=max_depth, max_features=max_features, max_leaf_nodes=max_leaf_nodes)
        if setting == 9:
            # TODO
            reg = RandomForestRegressor(n_estimators=n_estim, min_samples_leaf=30, max_depth=max_depth, max_features=max_features, max_leaf_nodes=max_leaf_nodes)
    elif regressor == 'mlp':
        hidden = int(kwargs["hidden"])
        layers = int(kwargs["layers"])
        reg = MlpRegressor(setting=setting, hidden_layers=(hidden, ) * layers)
    elif regressor == 'linear':
        if kwargs["interaction"] == "no":
            # no interaction
            reg = LinearRegression()
        elif kwargs["interaction"] == "yes":
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            Xtrain = poly.fit_transform(Xtrain)
            Xcalib = poly.fit_transform(Xcalib)
            Xtest = poly.fit_transform(Xtest)
            reg = LinearRegression()
        elif kwargs["interaction"] == "oracle":
            if setting in [1, 3, 5, 7]:
                transf = lambda x : np.column_stack((x, x[:, 0] * x[:, 1], x[:, 0] * x[:, 3], x[:, 1] * x[:, 3], x[:, 0] * x[:, 1] * x[:, 3]))
            if setting in [2, 4, 6, 8]:
                transf = lambda x : np.column_stack((x, x[:, 0] * x[:, 1]))
            Xtrain = transf(Xtrain)
            Xcalib = transf(Xcalib)
            Xtest = transf(Xtest)
            reg = LinearRegression()
    elif regressor == 'additive':
        if kwargs["interaction"] == "no":
            tm_list = TermList()
            for i in range(dim):
                tm_list += s(i)
            reg = LinearGAM(tm_list)
        elif kwargs["interaction"] == "yes":
            tm_list = TermList()
            for i in range(dim):
                tm_list += s(i)
                for j in range(i+1, dim):
                    tm_list += te(i, j)
            reg = LinearGAM(tm_list)
        elif kwargs["interaction"] == "oracle":
            tm_list = TermList()
            for i in range(4): # alternatively, use 4 here
                tm_list += s(i)
            if setting in [1, 3, 5, 7]:
                tm_list += te(0, 1, 3)
            if setting in [2, 4, 6, 8]:
                tm_list += te(0, 1)
            reg = LinearGAM(tm_list)
    elif regressor == 'oracle':
        reg = OracleRegressor2d_singledim(setting, d=0)
    
    if args.cont == 'False':
        # fit (Y > 0) directly, not Y
        reg.fit(Xtrain, 1 * (Ytrain[:, 0] > 0))
        
        # calibration 
        calib_scores = Ycalib[:, 0] - reg.predict(Xcalib)                          # BH_res
        calib_scores0 = - reg.predict(Xcalib)                                # BH_sub
        calib_scores_2clip = 1000 * (Ycalib[:, 0] > 0) - reg.predict(Xcalib)       # BH_clip (with M = 1000)
        
        Ypred = reg.predict(Xtest) 
        test_scores = -Ypred

        r_sq = r2_score((Ytest[:, 0] > 0), Ypred)
    elif args.cont == 'True':
        reg.fit(Xtrain, Ytrain[:, 0])
        # calibration 
        calib_scores = Ycalib[:, 0] - reg.predict(Xcalib)                          # BH_res
        calib_scores0 = - reg.predict(Xcalib)                                # BH_sub
        calib_scores_2clip = 1000 * (Ycalib[:, 0] > 0) - reg.predict(Xcalib)       # BH_clip (with M = 1000)
        
        Ypred = reg.predict(Xtest) 
        test_scores = -Ypred

        r_sq = r2_score(Ytest[:, 0], Ypred)

    # BH using residuals
    BH_res_1 = BH(calib_scores, test_scores, q )       
    # only use relevant samples to calibrate
    BH_rel_1 = BH(calib_scores0[Ycalib[:, 0] <= 0], test_scores, q ) 
    # use clipped scores
    BH_2clip_1 = BH(calib_scores_2clip, test_scores, q )
    # Bonferroni
    Bonf_1 = Bonferroni(calib_scores_2clip, test_scores, q )

    # for the second dimension:
    if regressor == 'rf':
        n_estim = int(kwargs["n_estim"])
        max_depth = int(kwargs["max_depth"])
        max_features = kwargs["max_features"]
        max_leaf_nodes = kwargs["max_leaf_nodes"]
        try:
            max_features = int(max_features)
        except ValueError:
            pass # input is 'sqrt' or 'log2'.
        try:
            max_leaf_nodes = int(max_leaf_nodes)
        except ValueError:
            max_leaf_nodes = None # no restriction

        reg = RfRegressor(setting=setting, n_estimators=n_estim, max_depth=max_depth, max_features=max_features, max_leaf_nodes=max_leaf_nodes)
        if setting == 9:
            # TODO
            reg = RandomForestRegressor(n_estimators=n_estim, min_samples_leaf=30, max_depth=max_depth, max_features=max_features, max_leaf_nodes=max_leaf_nodes)
    elif regressor == 'mlp':
        hidden = int(kwargs["hidden"])
        layers = int(kwargs["layers"])
        reg = MlpRegressor(setting=setting, hidden_layers=(hidden, ) * layers)
    elif regressor == 'linear':
        if kwargs["interaction"] == "no":
            # no interaction
            reg = LinearRegression()
        elif kwargs["interaction"] == "yes":
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            Xtrain = poly.fit_transform(Xtrain)
            Xcalib = poly.fit_transform(Xcalib)
            Xtest = poly.fit_transform(Xtest)
            reg = LinearRegression()
        elif kwargs["interaction"] == "oracle":
            if setting in [1, 3]:
                transf = lambda x : np.column_stack((x, x[:, 0] * x[:, 1], x[:, 0] * x[:, 3], x[:, 1] * x[:, 3], x[:, 0] * x[:, 1] * x[:, 3]))
            if setting in [2, 4]:
                transf = lambda x : np.column_stack((x, x[:, 0] * x[:, 1]))
            if setting in [5, 7]:
                transf = lambda x : np.column_stack((x, x[:, 0] * x[:, 1], x[:, 0] * x[:, 2], x[:, 1] * x[:, 2], x[:, 0] * x[:, 1] * x[:, 2]))
            if setting == 6:
                transf = lambda x : np.column_stack((x, x[:, 1] * x[:, 2]))
            if setting == 8:
                transf = lambda x : np.column_stack((x, x[:, 1] * x[:, 3]))
            Xtrain = transf(Xtrain)
            Xcalib = transf(Xcalib)
            Xtest = transf(Xtest)
            reg = LinearRegression()
    elif regressor == 'additive':
        if kwargs["interaction"] == "no":
            tm_list = TermList()
            for i in range(dim):
                tm_list += s(i)
            reg = LinearGAM(tm_list)
        elif kwargs["interaction"] == "yes":
            tm_list = TermList()
            for i in range(dim):
                tm_list += s(i)
                for j in range(i+1, dim):
                    tm_list += te(i, j)
            reg = LinearGAM(tm_list)
        elif kwargs["interaction"] == "oracle":
            tm_list = TermList()
            for i in range(4): # alternatively, use 4 here
                tm_list += s(i)
            if setting in [1, 3]:
                tm_list += te(0, 1, 3)
            if setting in [2, 4]:
                tm_list += te(0, 1)
            if setting in [5, 7]:
                tm_list += te(0, 1, 2)
            if setting == 6:
                tm_list += te(1, 2)
            if setting == 8:
                tm_list += te(1, 3)
            reg = LinearGAM(tm_list)
    elif regressor == 'oracle':
        reg = OracleRegressor2d_singledim(setting, d=1)
    
    if args.cont == 'False':
        # fit (Y > 0) directly, not Y
        reg.fit(Xtrain, 1 * (Ytrain[:, 1] > 0))
        
        # calibration 
        calib_scores = Ycalib[:, 1] - reg.predict(Xcalib)                          # BH_res
        calib_scores0 = - reg.predict(Xcalib)                                # BH_sub
        calib_scores_2clip = 1000 * (Ycalib[:, 1] > 0) - reg.predict(Xcalib)       # BH_clip (with M = 1000)
        
        Ypred = reg.predict(Xtest) 
        test_scores = -Ypred

        r_sq = r2_score((Ytest[:, 1] > 0), Ypred)
    elif args.cont == 'True':
        reg.fit(Xtrain, Ytrain[:, 1])
        # calibration 
        calib_scores = Ycalib[:, 1] - reg.predict(Xcalib)                          # BH_res
        calib_scores0 = - reg.predict(Xcalib)                                # BH_sub
        calib_scores_2clip = 1000 * (Ycalib[:, 1] > 0) - reg.predict(Xcalib)       # BH_clip (with M = 1000)
        
        Ypred = reg.predict(Xtest) 
        test_scores = -Ypred

        r_sq = r2_score(Ytest[:, 1], Ypred)

    # BH using residuals
    BH_res_2 = BH(calib_scores, test_scores, q )
    # only use relevant samples to calibrate
    BH_rel_2 = BH(calib_scores0[Ycalib[:, 1] <= 0], test_scores, q )   
    # use clipped scores
    BH_2clip_2 = BH(calib_scores_2clip, test_scores, q )
    # Bonferroni
    Bonf_2 = Bonferroni(calib_scores_2clip, test_scores, q )

    # now, intersect and calculate the metrics
    BH_res = [i for i in BH_res_1 if i in set(BH_res_2)]
    BH_rel = [i for i in BH_rel_1 if i in set(BH_rel_2)]
    BH_2clip = [i for i in BH_2clip_1 if i in set(BH_2clip_2)]
    Bonf = [i for i in Bonf_1 if i in set(Bonf_2)]

    if len(BH_res) == 0:
        BH_res_fdp = 0
        BH_res_power = 0
    else:
        BH_res_fdp = np.sum((Ytest[BH_res][:, 0] <= 0) | (Ytest[BH_res][:, 1] <= 0)) / len(BH_res)
        BH_res_power = np.sum((Ytest[BH_res][:, 0] > 0) & (Ytest[BH_res][:, 1] > 0)) / true_null if true_null != 0 else 0

    if len(BH_rel) == 0:
        BH_rel_fdp = 0
        BH_rel_power = 0
    else:
        BH_rel_fdp = np.sum((Ytest[BH_rel][:, 0] <= 0) | (Ytest[BH_rel][:, 1] <= 0)) / len(BH_rel)
        BH_rel_power = np.sum((Ytest[BH_rel][:, 0] > 0) & (Ytest[BH_rel][:, 1] > 0)) / true_null if true_null != 0 else 0
    
    if len(BH_2clip) == 0:
        BH_2clip_fdp = 0
        BH_2clip_power = 0
    else:
        BH_2clip_fdp = np.sum((Ytest[BH_2clip][:, 0] <= 0) | (Ytest[BH_2clip][:, 1] <= 0)) / len(BH_2clip)
        BH_2clip_power = np.sum((Ytest[BH_2clip][:, 0] > 0) & (Ytest[BH_2clip][:, 1] > 0)) / true_null if true_null != 0 else 0

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
            'regressor': [regressor],
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
    sig, covar, setting, seed, x = tup

    if regressor == 'rf':
        n_estim = x if xaxis == 'n_estim' else config["n_estim"]
        max_depth = x if xaxis == 'max_depth' else config["max_depth"]
        max_features = x if xaxis == 'max_features' else config["max_features"]
        return run(sig, covar, setting, seed, n_estim=n_estim, max_depth=max_depth, max_features=max_features)
    elif regressor == 'mlp':
        hidden = x if xaxis == 'hidden' else config["hidden"]
        layers = x if xaxis == 'layers' else config["layers"]
        return run(sig, covar, setting, seed, hidden=hidden, layers=layers)
    elif regressor in ['linear', 'additive']:
        return run(sig, covar, setting, seed, interaction=x)
    elif regressor == 'oracle':
        return run(sig, covar, setting, seed)

if __name__ == '__main__':
    if regressor in ['rf', 'mlp']:
        combined_itr = itertools.product(sig_list, covar_list, set_list, seed_list, range(*xrange))
        combined_itr2 = itertools.product(sig_list2, covar_list2, set_list2, seed_list, range(*xrange))
        total_len = len(sig_list) * len(covar_list) * len(set_list) * len(seed_list) * len(range(*xrange))
        total_len2 = len(sig_list2) * len(covar_list2) * len(set_list2) * len(seed_list) * len(range(*xrange))
    elif regressor in ['linear', 'additive']:
        combined_itr = itertools.product(sig_list, covar_list, set_list, seed_list, [interaction])
        combined_itr2 = itertools.product(sig_list2, covar_list2, set_list2, seed_list, [interaction])
        total_len = len(sig_list) * len(covar_list) * len(set_list) * len(seed_list)
        total_len2 = len(sig_list2) * len(covar_list2) * len(set_list2) * len(seed_list)
    elif regressor == 'oracle':
        combined_itr = itertools.product(sig_list, covar_list, set_list, seed_list, [None])
        combined_itr2 = itertools.product(sig_list2, covar_list2, set_list2, seed_list, [None])
        total_len = len(sig_list) * len(covar_list) * len(set_list) * len(seed_list)
        total_len2 = len(sig_list2) * len(covar_list2) * len(set_list2) * len(seed_list)

    with Pool(processes=6) as pool:
        results = list(tqdm(pool.imap(run2, combined_itr), total=total_len))
    with Pool(processes=6) as pool:
        results2 = list(tqdm(pool.imap(run2, combined_itr2), total=total_len2))

    all_res = pd.concat(results, ignore_index=True)
    all_res2 = pd.concat(results2, ignore_index=True)
    all_res = pd.concat((all_res, all_res2), ignore_index=True)
                        
    all_res.to_csv(full_out_dir) 