import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn

class OracleRegressor():
    def __init__(self, setting: int):
        assert 1 <= setting and setting <= 8
        self.s = setting
        self.model = LinearRegression()

    def extract(self, X):
        if self.s == 1:
            U1 = (X[:,0] * X[:,1] > 0) * (X[:,3] * (X[:,3] > 0.5) + 0.5 * (X[:,3] <= 0.5))
            U2 = (X[:,0] * X[:,1] <= 0) * (X[:,3] * (X[:,3] < -0.5) - 0.5 * (X[:,3] > -0.5))
            return np.column_stack((U1, U2))
        
        if 2 <= self.s and self.s <= 4:
            U1 = X[:,0] * X[:,1]
            U2 = np.exp(X[:,3] - 1)
            return np.column_stack((U1, U2))

        if self.s == 5:
            U1 = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3])
            U2 = (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
            return np.column_stack((U1, U2))
        
        if 6 <= self.s and self.s <= 8:
            U1 = X[:,0] * X[:,1]
            U2 = X[:,2] ** 2
            U3 = np.exp(X[:,3] - 1)
            return np.column_stack((U1, U2, U3))

    def fit(self, X, Y):
        self.model.fit(self.extract(X), Y)

    def predict(self, X):
        return self.model.predict(self.extract(X))
    

# for deep learning quantile regressors, need to write manually with torch
''' 
alpha: lower quantile
dropout_p: dropout probability in every layer
'''
class MLPQuantileRegressor(nn.Module):
    def __init__(self, alpha, in_channel, out_channel, layers=[16, 16, 16], dropout_p=0.1):
        super().__init__()
        self.layers = layers

        dims = [in_channel] + layers + [out_channel]
        modules = []
        for i in range(len(dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    nn.Dropout(p=dropout_p), # regularization
                    nn.ReLU(),
                )
            )
        self.mlp = nn.Sequential(*modules)

    def forward(self, input):
        return self.mlp(input)
    
    def loss(self, y, q_pred):
        error = y - q_pred
        # pinball loss
        return torch.mean(torch.max(self.alpha * error, (self.alpha - 1) * error))


def gen_data(setting, n, sig, dim=20):     
    if setting == 0:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = X[:,0] * 2 + X[:, 1] * 3 + X[:, 2] * (-2)
        Y = mu_x + np.random.normal(size=n) * sig
        return X, Y, mu_x

    if setting == 1: 
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] > 0) * (X[:,3] * (X[:,3] > 0.5) + 0.5 * (X[:,3] <= 0.5)) + (X[:,0] * X[:,1] <= 0) * (X[:,3] * (X[:,3] < -0.5) - 0.5 * (X[:,3] > -0.5))
        mu_x = mu_x * 4
        Y = mu_x + np.random.normal(size=n) * sig
        # plt.scatter(mu_x, Y)
        return X, Y, mu_x
    
    if setting == 2:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        Y = mu_x + np.random.normal(size=n) * 1.5 * sig 
        return X, Y, mu_x
    
    if setting == 3:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        Y = mu_x + np.random.normal(size=n) * (5.5 - abs(mu_x))/2 * sig 
        return X, Y, mu_x
    
    if setting == 4:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        sig_x = 0.25 * mu_x**2 * (np.abs(mu_x) < 2) + 0.5 * np.abs(mu_x) * (np.abs(mu_x) >= 1)
        Y = mu_x + np.random.normal(size=n) * sig_x * sig
        return X, Y, mu_x
    
    if setting == 5:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
        mu_x = mu_x  
        Y = mu_x + np.random.normal(size=n) * sig
        return X, Y, mu_x
    
    if setting == 6:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
        Y = mu_x + np.random.normal(size=n) * 1.5 * sig 
        return X, Y, mu_x
    
    if setting == 7:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
        Y = mu_x + np.random.normal(size=n) * (5.5 - abs(mu_x))/2 * sig 
        return X, Y, mu_x
    
    if setting == 8:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
        sig_x = 0.25 * mu_x**2 * (np.abs(mu_x) < 2) + 0.5 * np.abs(mu_x) * (np.abs(mu_x) >= 1)
        Y = mu_x + np.random.normal(size=n) * sig_x * sig
        return X, Y, mu_x

    if setting == 9: # from CQR paper
        # in the original paper, feature is 1-dimensional. Here we choose to only use the first covariate, X[:, 0].
        X = np.random.uniform(low=0, high=5, size=n*dim).reshape((n,dim))
        Y = np.zeros(n)
        for i in range(n):
            Y[i] = np.random.poisson(np.sin(X[i, 0]) ** 2 + 0.1) + 0.03 * X[i, 0] * np.random.randn(1) - 3 + 2 * X[i, 0] # shift 2
            Y[i] += 25 * (np.random.uniform(0, 1, 1) < 0.01) * np.random.randn(1)
        return X, Y, None

def gen_data_2d(setting, n, sig, dim=20):
    X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
    # generate a multivariate case
    if setting == 1:
        mu_x1 = (X[:,0] * X[:,1] > 0) * (X[:,3] * (X[:,3] > 0.5) + 0.5 * (X[:,3] <= 0.5)) + (X[:,0] * X[:,1] <= 0) * (X[:,3] * (X[:,3] < -0.5) - 0.5 * (X[:,3] > -0.5))
        mu_x2 = (X[:,1] * X[:,2] > 0) * (X[:,0] * (X[:,0] > 0.5) + 0.5 * (X[:,2] <= 0.5)) + (X[:,1] * X[:,2] <= 0) * (X[:,2] * (X[:,1] < -0.5) - 0.5 * (X[:,1] > -0.5))
        mean = np.column_stack((mu_x1, mu_x2))
        cov = [[      sig, 0 * sig],
               [0 * sig,       sig]]
        rng = np.random.default_rng(33)
        Y = mean + rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
        return X, Y, mu_x1, mu_x2, cov
    
    if setting == 2:
        mu_x1 = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        mu_x2 = (X[:,1] * X[:,2] + np.exp(X[:,0] - 1)) * 5
        mean = np.column_stack((mu_x1, mu_x2))
        cov = [[      sig, 0 * sig],
               [0 * sig,       sig]]
        rng = np.random.default_rng(33)
        Y = mean + rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
        return X, Y, mu_x1, mu_x2, cov
    
    if setting == 5:
        mu_x1 = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
        mu_x2 = (X[:,2] * X[:,1] > 0) * (X[:,0] > 0.5) * (0.25 + X[:,0]) + (X[:,2] * X[:,1] <= 0) * (X[:,0] < -0.5) * (X[:,0] - 0.25)
        mean = np.column_stack((mu_x1, mu_x2))
        cov = [[      sig, 0 * sig],
               [0 * sig,       sig]]
        rng = np.random.default_rng(33)
        Y = mean + rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
        return X, Y, mu_x1, mu_x2, cov
    
    if setting == 6:
        mu_x1 = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
        mu_x2 = (X[:,3] * X[:,1] + X[:,0] ** 2 + np.exp(X[:,2] - 1) - 1) * 2
        mean = np.column_stack((mu_x1, mu_x2))
        cov = [[      sig, 0 * sig],
               [0 * sig,       sig]]
        rng = np.random.default_rng(33)
        Y = mean + rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
        return X, Y, mu_x1, mu_x2, cov

def BH(calib_scores, test_scores, q = 0.1, return_pval=False):
    ntest = len(test_scores)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)
    
    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (ncalib+1)
         
    # BH(q) 
    df_test = pd.DataFrame({"id": range(ntest), "pval": pvals}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    
    if len(idx_smaller) == 0:
        if not return_pval:
            return (np.array([]))
        else:
            return np.array([]), pvals
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller)+1)])
        if not return_pval:
            return (idx_sel)
        else:
            return idx_sel, pvals
    
def Bonferroni(calib_scores, test_scores, q = 0.1):
    ntest = len(test_scores)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)
    
    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (ncalib+1)

    idxs = [j for j in range(ntest) if pvals[j] <= q / ntest]
    return np.array(idxs)
