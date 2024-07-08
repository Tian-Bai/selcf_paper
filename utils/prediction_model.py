import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn

''' 
The true models.
'''
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
    
class OracleRegressor2d():
    def __init__(self, setting: int):
        assert setting in [1, 2, 5, 6]
        self.s = setting
        self.model = LinearRegression()

    def extract(self, X):
        if self.s == 1:
            U1 = (X[:,0] * X[:,1] > 0) * (X[:,3] * (X[:,3] > 0.5) + 0.5 * (X[:,3] <= 0.5))
            U2 = (X[:,0] * X[:,1] <= 0) * (X[:,3] * (X[:,3] < -0.5) - 0.5 * (X[:,3] > -0.5))
            U3 = (X[:,1] * X[:,2] > 0) * (X[:,0] * (X[:,0] > 0.5) + 0.5 * (X[:,2] <= 0.5))
            U4 = (X[:,1] * X[:,2] <= 0) * (X[:,2] * (X[:,1] < -0.5) - 0.5 * (X[:,1] > -0.5))
            return np.column_stack((U1, U2, U3, U4))
        
        if 2 <= self.s and self.s <= 4:
            U1 = X[:,0] * X[:,1]
            U2 = np.exp(X[:,3] - 1)
            U3 = X[:,1] * X[:,2]
            U4 = np.exp(X[:,0] - 1)
            return np.column_stack((U1, U2, U3, U4))

        if self.s == 5:
            U1 = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3])
            U2 = (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
            U3 = (X[:,2] * X[:,1] > 0) * (X[:,0] > 0.5) * (0.25 + X[:,0]) 
            U4 = (X[:,2] * X[:,1] <= 0) * (X[:,0] < -0.5) * (X[:,0] - 0.25)
            return np.column_stack((U1, U2, U3, U4))
        
        if 6 <= self.s and self.s <= 8:
            U1 = X[:,0] * X[:,1]
            U2 = X[:,2] ** 2
            U3 = np.exp(X[:,3] - 1)
            U4 = X[:,3] * X[:,1]
            U5 = X[:,0] ** 2
            U6 = np.exp(X[:,2] - 1)
            return np.column_stack((U1, U2, U3, U4, U5, U6))

    def fit(self, X, Y):
        self.model.fit(self.extract(X), Y)

    def predict(self, X):
        return self.model.predict(self.extract(X))
    
''' 
Wrapper classes for summarizing the best hyperparameters.
'''
class RfRegressor():
    def __init__(self, setting: int, n_estimators=50, max_depth=20, max_features=10):
        assert 1 <= setting <= 8
        if setting != 5:
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=0)
        else:
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_leaf=20, min_samples_split=20, random_state=0)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
class MlpRegressor():
    def __init__(self, setting: int, hidden_layers=(32, ) * 4):
        assert 1 <= setting <= 8
        if setting == 1:
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, random_state=0, alpha=0.5, max_iter=1000, learning_rate_init=1e-4, early_stopping=True, tol=1e-6)
        if setting in [2, 3, 4]:
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, random_state=0, alpha=0.5, max_iter=1000, learning_rate_init=1e-4, early_stopping=True, tol=1e-6)
        if setting == 5:
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, random_state=0, alpha=2, max_iter=1000, learning_rate_init=1e-4, early_stopping=True, tol=1e-6)
        if setting in [6, 7, 8]:
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, random_state=0, alpha=0.2, max_iter=1000, learning_rate_init=1e-4, early_stopping=True, tol=1e-6)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    