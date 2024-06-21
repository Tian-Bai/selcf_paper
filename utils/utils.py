#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:54:03 2022

@author: ying
"""

import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression

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


def gen_data(setting, n, sig, dim=20): 
    X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
    
    if setting == 1: 
        mu_x = (X[:,0] * X[:,1] > 0) * (X[:,3] * (X[:,3] > 0.5) + 0.5 * (X[:,3] <= 0.5)) + (X[:,0] * X[:,1] <= 0) * (X[:,3] * (X[:,3] < -0.5) - 0.5 * (X[:,3] > -0.5))
        mu_x = mu_x * 4
        Y = mu_x + np.random.normal(size=n) * sig
        # plt.scatter(mu_x, Y)
        return X, Y, mu_x
    
    if setting == 2:
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        Y = mu_x + np.random.normal(size=n) * 1.5 * sig 
        return X, Y, mu_x
    
    if setting == 3:
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        Y = mu_x + np.random.normal(size=n) * (5.5 - abs(mu_x))/2 * sig 
        return X, Y, mu_x
    
    if setting == 4:
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        sig_x = 0.25 * mu_x**2 * (np.abs(mu_x) < 2) + 0.5 * np.abs(mu_x) * (np.abs(mu_x) >= 1)
        Y = mu_x + np.random.normal(size=n) * sig_x * sig
        return X, Y, mu_x
    
    if setting == 5:
        mu_x = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
        mu_x = mu_x  
        Y = mu_x + np.random.normal(size=n) * sig
        return X, Y, mu_x
    
    if setting == 6:
        mu_x = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
        Y = mu_x + np.random.normal(size=n) * 1.5 * sig 
        return X, Y, mu_x
    
    if setting == 7:
        mu_x = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
        Y = mu_x + np.random.normal(size=n) * (5.5 - abs(mu_x))/2 * sig 
        return X, Y, mu_x
    
    if setting == 8:
        mu_x = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
        sig_x = 0.25 * mu_x**2 * (np.abs(mu_x) < 2) + 0.5 * np.abs(mu_x) * (np.abs(mu_x) >= 1)
        Y = mu_x + np.random.normal(size=n) * sig_x * sig
        return X, Y, mu_x

def BH(calib_scores, test_scores, q = 0.1):
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
        return(np.array([]))
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller)+1)])
        return(idx_sel)
    
def Bonferroni(calib_scores, test_scores, q = 0.1):
    ntest = len(test_scores)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)
    
    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (ncalib+1)

    idxs = [j for j in range(ntest) if pvals[j] <= q / ntest]
    return np.array(idxs)
