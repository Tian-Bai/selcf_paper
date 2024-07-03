from DeepPurpose import utils, dataset, CompoundPred
import warnings
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm
import contextlib
from rdkit import rdBase
import argparse
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM, s, te
from pygam.terms import TermList
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from multiprocessing import Pool
from sklearn.metrics import r2_score

rf_param = ['n_estim', 'max_depth', 'max_features']
mlp_param = ['hidden', 'layers']

def rf_config(value):
    try:
        pairs = {}
        for pair in value.split(','):
            k, v = pair.split(':')
            assert k in rf_param
            pairs[k.strip()] = v.strip()
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            'Illegal argument for rf configurations.'
        )
    return pairs

def mlp_config(value):
    try:
        pairs = {}
        for pair in value.split(','):
            k, v = pair.split(':')
            assert k in mlp_param
            pairs[k.strip()] = v.strip()
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            'Illegal argument for mlp configurations.'
        )
    return pairs

def interaction_type(value):
    try:
        s = str(value).lower()
        assert s in ['yes', 'y', 'no', 'n', 'oracle', 'o']
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            'Illegal argument for linear model type. Should be either "yes", "no", or "oracle".'
        )
    if s == 'y':
        s = 'yes'
    elif s == 'n':
        s = 'no'
    elif s == 'o':
        s = 'oracle'
    return s


parser = argparse.ArgumentParser(description='Select regressor and configurations.')
parser.add_argument('-i', '--input', dest='itr', type=int, help='number of tests (seeds)', default=100)

subparsers = parser.add_subparsers(dest='regressor', required=True, help='The target regressor. Choose between ["rf", "mlp", "additive", "linear", ...].')
parser_rf = subparsers.add_parser('rf', help='rf regressor parser.')
parser_mlp = subparsers.add_parser('mlp', help='mlp regressor parser.')
parser_linear = subparsers.add_parser('linear', help='linear regressor parser.')
parser_additive = subparsers.add_parser('additive', help='GAM regressor parser.')
parser_mlp_dp = subparsers.add_parser('mlp_DP', help='mlp-DP regressor parser.')

# for below two regressors, rf and mlp, we allow testing along an x axis representing the configuration of models (e.g. number of hidden layers, ...)
# rf parser
parser_rf.add_argument('config', type=rf_config, help='other configurations of the rf') 

# mlp parser
parser_mlp.add_argument('config', type=mlp_config, help='other configurations of the mlp')

# for below two regressors, linear and additive, we disable the choise of having interactions since there will be too many terms (1024 x 1024)
args = parser.parse_args()

warnings.filterwarnings("ignore")
rdBase.DisableLog('rdApp.warning')

@contextlib.contextmanager
def suppress_print():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield

def single_select(calib_scores, test_scores, q = 0.1):
    ntest = len(test_scores)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)

    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (ncalib+1)

    idxs = [j for j in range(ntest) if pvals[j] <= q]
    return np.array(idxs), pvals

def bonf_select(calib_scores, test_scores, q = 0.1):
    ntest = len(test_scores)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)

    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (ncalib+1)

    idxs = [j for j in range(ntest) if pvals[j] <= q / ntest]
    return np.array(idxs), pvals

def conformal_select(calib_scores, test_scores, q = 0.1):
    ntest = len(test_scores)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)

    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (ncalib+1)
    
    # BH(q) 
    df_test = pd.DataFrame({"id": range(ntest), "pval": pvals, "scores": test_scores}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,3]]
     
    if len(idx_smaller) == 0:
        return np.array([]), pvals
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller))])
        s_th = df_test.iloc[idx_smaller, 3]
        return idx_sel, pvals

def eval_sel(sel_idx, ys, cs):
    if len(sel_idx) == 0:
        fdp = 0
        power = 0
    else:
        fdp = np.sum(ys[sel_idx] <= cs[sel_idx]) / len(sel_idx)
        power = np.sum(ys[sel_idx] > cs[sel_idx]) / sum(ys > cs) 
    return fdp, power, len(sel_idx)

if args.regressor in ['linear', 'additive']:
    out_dir = f"..\\csv-HIV\\{args.regressor}"
    full_out_dir = f"..\\csv-HIV\\{args.regressor}\\itr={args.itr}.csv"
elif args.regressor == 'rf':
    config = args.config
    out_dir = f"..\\csv-HIV\\{args.regressor}"
    full_out_dir = f"..\\csv-HIV\\{args.regressor}\\{rf_param[0]}={config[rf_param[0]]} {rf_param[1]}={config[rf_param[1]]} {rf_param[2]}={config[rf_param[2]]} itr={args.itr}.csv"
elif args.regressor == 'mlp':
    config = args.config
    out_dir = f"..\\csv-HIV\\{args.regressor}"
    full_out_dir = f"..\\csv-HIV\\{args.regressor}\\{mlp_param[0]}={config[mlp_param[0]]} {mlp_param[1]}={config[mlp_param[1]]} itr={args.itr}.csv"
elif args.regressor == 'mlp_DP':
    out_dir = f"..\\csv-HIV\\{args.regressor}"
    full_out_dir = f"..\\csv-HIV\\{args.regressor}\\itr={args.itr}.csv"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print("Output diretory created!")

X_drugs, y, drugs_index = dataset.load_HIV(path = './data')
drug_encoding = 'Morgan'

n = len(y)

seed_list = [i for i in range(0, args.itr)]

def run(seed):
    reind = np.random.permutation(n)
    X_drugs_train = X_drugs[reind[0:int(n*0.4+1)]]
    y_train = y[reind[0:int(n*0.4+1)]]
    X_drugs_other = X_drugs[reind[int(1+n*0.4):n]]
    y_other = y[reind[int(1+n*0.4):n]]

    with suppress_print():
        ttrain, tval, ttest = utils.data_process(X_drug = X_drugs_train, y = y_train, 
                                                drug_encoding = drug_encoding,
                                                split_method='random', frac=[0.7, 0.1, 0.2],
                                                random_seed = seed)

        dother = utils.data_process(X_drug = X_drugs_other, y = y_other, 
                                            drug_encoding = drug_encoding,
                                            split_method='no_split',
                                            random_seed = seed)
        
    # USING CUSTOM PREDICTOR
    ttrain_label = ttrain.Label.to_numpy()
    ttrain_predictor = np.stack(ttrain['drug_encoding'].to_numpy()) 
    dother_predictor = np.stack(dother['drug_encoding'].to_numpy())

    if args.regressor == 'linear':
        model = LinearRegression()
        model.fit(ttrain_predictor, ttrain_label)
        all_pred = model.predict(dother_predictor)
        model.fit(ttrain_predictor, ttrain_label)
        all_pred = model.predict(dother_predictor)
    elif args.regressor == 'additive':
        # package problem now
        dim = len(ttrain_predictor[0])
        tm_list = TermList()
        for i in range(dim):
            tm_list += s(i)
        model = LinearGAM(tm_list)
        model.fit(ttrain_predictor, ttrain_label)
        all_pred = model.predict(dother_predictor)
    elif args.regressor == 'mlp':
        hidden = int(args.config["hidden"])
        layers = int(args.config["layers"])
        model = MLPRegressor(hidden_layer_sizes=(hidden, ) * layers, random_state=0, alpha=3e-3, max_iter=1000)
        model.fit(ttrain_predictor, ttrain_label)
        all_pred = model.predict(dother_predictor)
    elif args.regressor == 'rf':
        n_estim = int(args.config["n_estim"])
        max_depth = int(args.config["max_depth"])
        max_features = args.config["max_features"]
        model = RandomForestRegressor(n_estimators=n_estim, max_depth=max_depth, max_features=max_features, random_state=0)
        model.fit(ttrain_predictor, ttrain_label)
        all_pred = model.predict(dother_predictor)
    elif args.regressor == 'mlp_DP':
        with suppress_print():
            model_config = utils.generate_config(drug_encoding = drug_encoding, 
                                cls_hidden_dims = [1024, 1024, 512], 
                                train_epoch = 3, 
                                LR = 0.001, 
                                batch_size = 128,
                                hidden_dim_drug = 128,
                                mpnn_hidden_size = 128,
                                mpnn_depth = 3
                                )
            model = CompoundPred.model_initialize(**model_config)
            model.train(ttrain, tval, ttest)

            all_pred = np.array(model.predict(dother))
            train_pred = np.array(model.predict(ttrain))

    if args.regressor != 'mlp_DP':
        ttest_label = ttest.Label.to_numpy()
        ttest_predictor = np.stack(ttest['drug_encoding'].to_numpy()) 
        ttest_pred = model.predict(ttest_predictor)
        r_sq = r2_score(ttest_label, ttest_pred)
    else:
        ttest_label = ttest.Label.to_numpy()
        ttest_pred = np.array(model.predict(ttest))
        r_sq = r2_score(ttest_label, ttest_pred)

    reind = np.random.permutation(len(dother))

    dcalib = dother.iloc[reind[:int(len(dother)*0.5+1)]]
    dtest = dother.iloc[reind[int(1+len(dother)*0.5):]]

    hat_mu_calib = all_pred[reind[:int(len(dother)*0.5+1)]]
    hat_mu_test = all_pred[reind[int(1+len(dother)*0.5):]]

    y_calib = np.array(dcalib["Label"])
    y_test = np.array(dtest["Label"])

    c = 0

    calib_scores_res = y_calib - hat_mu_calib
    calib_scores_sub = - hat_mu_calib 
    calib_scores_clip = 100 * (y_calib > c) + c * (y_calib <= c) - hat_mu_calib

    test_scores = c - hat_mu_test

    q = 0.1 # nominal level

    single, _ = single_select(calib_scores_clip, test_scores, q)
    bonf, _ = bonf_select(calib_scores_clip, test_scores, q)
    BH_res, _ = conformal_select(calib_scores_res, test_scores, q)  
    BH_sub, _ = conformal_select(calib_scores_sub[y_calib <= c], test_scores, q) 
    BH_clip, _ = conformal_select(calib_scores_clip, test_scores, q)

    single_fdp, single_power, single_nsel = eval_sel(single, y_test, np.array([c]*len(y_test)))
    bonf_fdp, bonf_power, bonf_nsel = eval_sel(bonf, y_test, np.array([c]*len(y_test)))
    BH_res_fdp, BH_res_power, BH_res_nsel = eval_sel(BH_res, y_test, np.array([c]*len(y_test)))
    BH_sub_fdp, BH_sub_power, BH_sub_nsel = eval_sel(BH_sub, y_test, np.array([c]*len(y_test)))
    BH_clip_fdp, BH_clip_power, BH_clip_nsel = eval_sel(BH_clip, y_test, np.array([c]*len(y_test))) 

    output_dict = {
        'q': [q], 
        'seed': [seed], 
        'regressor': [args.regressor],
        'r_squared': [r_sq],
        'calib_size': [len(y_calib)], 
        'test_size': [len(y_test)], 
        'single_power': [single_power],
        'single_fdp': [single_fdp],
        'single_nsel': [single_nsel], 
        'bonf_power': [bonf_power],
        'bonf_fdp': [bonf_fdp],
        'bonf_nsel': [bonf_nsel],
        'BH_res_fdp': [BH_res_fdp], 
        'BH_res_power': [BH_res_power], 
        'BH_res_nsel': [BH_res_nsel], 
        'BH_sub_fdp': [BH_sub_fdp], 
        'BH_sub_power': [BH_sub_power], 
        'BH_sub_nsel': [BH_sub_nsel], 
        'BH_clip_fdp': [BH_clip_fdp], 
        'BH_clip_power': [BH_clip_power],
        'BH_clip_nsel': [BH_clip_nsel]
    }
    df = pd.DataFrame(output_dict)
    return df

if __name__ == '__main__':
    combined_itr = seed_list
    total_len = len(seed_list)

    with Pool(processes=6) as pool:
        results = list(tqdm(pool.imap(run, combined_itr), total=total_len))

    all_res = pd.concat(results, ignore_index=True)
    all_res.to_csv(full_out_dir)
