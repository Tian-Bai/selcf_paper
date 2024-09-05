import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

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
    
class OracleRegressor2d_singledim():
    def __init__(self, setting: int, d):
        assert 1 <= setting and setting <= 8
        self.s = setting
        self.d = d
        self.model = LinearRegression()

    def extract(self, X):
        if self.s == 1 or (self.s == 5 and self.d == 0):
            U1 = (X[:,0] * X[:,1] > 0) * (X[:,3] * (X[:,3] > 0.5) + 0.5 * (X[:,3] <= 0.5))
            U2 = (X[:,0] * X[:,1] <= 0) * (X[:,3] * (X[:,3] < -0.5) - 0.5 * (X[:,3] > -0.5))
            return np.column_stack((U1, U2))
        
        if self.s == 2 or (self.s == 6 and self.d == 0):
            U1 = X[:,0] * X[:,1]
            U2 = np.exp(X[:,3] - 1)
            return np.column_stack((U1, U2))

        if self.s == 3 or (self.s == 7 and self.d == 0):
            U1 = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3])
            U2 = (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
            return np.column_stack((U1, U2))
        
        if self.s == 4 or (self.s == 8 and self.d == 0):
            U1 = X[:,0] * X[:,1]
            U2 = X[:,2] ** 2
            U3 = np.exp(X[:,3] - 1)
            return np.column_stack((U1, U2, U3))

        if self.s == 5 and self.d == 1:
            U1 = (X[:,1] * X[:,2] > 0) * (X[:,0] * (X[:,0] > 0.5) + 0.5 * (X[:,0] <= 0.5)) 
            U2 = (X[:,1] * X[:,2] <= 0) * (X[:,0] * (X[:,0] < -0.5) - 0.5 * (X[:,0] > -0.5))
            return np.column_stack((U1, U2))
        
        if self.s == 6 and self.d == 1:
            U1 = (X[:,1] * X[:,2])
            U2 = np.exp(X[:,0] - 1)
            return np.column_stack((U1, U2))
        
        if self.s == 7 and self.d == 1:
            U1 = (X[:,2] * X[:,1] > 0) * (X[:,0] > 0.5) * (0.25 + X[:,0]) 
            U2 = (X[:,2] * X[:,1] <= 0) * (X[:,0] < -0.5) * (X[:,0] - 0.25)
            return np.column_stack((U1, U2))
        
        if self.s == 8 and self.d == 1:
            U1 = (X[:,3] * X[:,1])
            U2 = X[:,0] ** 2
            U3 = np.exp(X[:,2] - 1) - 1
            return np.column_stack((U1, U2, U3))

    def fit(self, X, Y):
        self.model.fit(self.extract(X), Y)

    def predict(self, X):
        return self.model.predict(self.extract(X))

class OracleRegressor2d():
    def __init__(self, setting: int):
        assert 1 <= setting and setting <= 8
        self.s = setting
        self.model = LinearRegression()

    def extract(self, X):
        if self.s == 1:
            U1 = (X[:,0] * X[:,1] > 0) * (X[:,3] * (X[:,3] > 0.5) + 0.5 * (X[:,3] <= 0.5))
            U2 = (X[:,0] * X[:,1] <= 0) * (X[:,3] * (X[:,3] < -0.5) - 0.5 * (X[:,3] > -0.5))
            return np.column_stack((U1, U2))
        
        if self.s == 2:
            U1 = X[:,0] * X[:,1]
            U2 = np.exp(X[:,3] - 1)
            return np.column_stack((U1, U2))

        if self.s == 3:
            U1 = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3])
            U2 = (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
            return np.column_stack((U1, U2))
        
        if self.s == 4:
            U1 = X[:,0] * X[:,1]
            U2 = X[:,2] ** 2
            U3 = np.exp(X[:,3] - 1)
            return np.column_stack((U1, U2, U3))
        
        if self.s == 5:
            U1 = (X[:,0] * X[:,1] > 0) * (X[:,3] * (X[:,3] > 0.5) + 0.5 * (X[:,3] <= 0.5))
            U2 = (X[:,0] * X[:,1] <= 0) * (X[:,3] * (X[:,3] < -0.5) - 0.5 * (X[:,3] > -0.5))
            U3 = (X[:,1] * X[:,2] > 0) * (X[:,0] * (X[:,0] > 0.5) + 0.5 * (X[:,0] <= 0.5))
            U4 = (X[:,1] * X[:,2] <= 0) * (X[:,0] * (X[:,0] < -0.5) - 0.5 * (X[:,0] > -0.5))
            return np.column_stack((U1, U2, U3, U4))
        
        if self.s == 6:
            U1 = X[:,0] * X[:,1]
            U2 = np.exp(X[:,3] - 1)
            U3 = X[:,1] * X[:,2]
            U4 = np.exp(X[:,0] - 1)
            return np.column_stack((U1, U2, U3, U4))

        if self.s == 7:
            U1 = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3])
            U2 = (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
            U3 = (X[:,2] * X[:,1] > 0) * (X[:,0] > 0.5) * (0.25 + X[:,0]) 
            U4 = (X[:,2] * X[:,1] <= 0) * (X[:,0] < -0.5) * (X[:,0] - 0.25)
            return np.column_stack((U1, U2, U3, U4))
        
        if self.s == 8:
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
    
class TrueRegressor2d():
    def __init__(self, setting: int):
        assert setting in [1, 2, 3, 4]
        self.s = setting

    def fit(self, X, Y):
        pass

    def predict(self, X):
        if self.s == 1:
            mu_x1 = (X[:,0] * X[:,1] > 0) * (X[:,3] * (X[:,3] > 0.5) + 0.5 * (X[:,3] <= 0.5)) + (X[:,0] * X[:,1] <= 0) * (X[:,3] * (X[:,3] < -0.5) - 0.5 * (X[:,3] > -0.5))
            # mu_x2 = (X[:,1] * X[:,2] > 0) * (X[:,0] * (X[:,0] > 0.5) + 0.5 * (X[:,2] <= 0.5)) + (X[:,1] * X[:,2] <= 0) * (X[:,2] * (X[:,1] < -0.5) - 0.5 * (X[:,1] > -0.5))
            mu_x2 = (X[:,0] * X[:,1] > 0) * (X[:,3] * (X[:,3] > 0.5) + 0.5 * (X[:,3] <= 0.5)) + (X[:,0] * X[:,1] <= 0) * (X[:,3] * (X[:,3] < -0.5) - 0.5 * (X[:,3] > -0.5))
            return np.column_stack((mu_x1, mu_x2))
        
        if self.s == 2:
            mu_x1 = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
            mu_x2 = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
            # mu_x2 = (X[:,1] * X[:,2] + np.exp(X[:,0] - 1)) * 5
            return np.column_stack((mu_x1, mu_x2))
        
        if self.s == 3:
            mu_x1 = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
            mu_x2 = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
            # mu_x2 = (X[:,2] * X[:,1] > 0) * (X[:,0] > 0.5) * (0.25 + X[:,0]) + (X[:,2] * X[:,1] <= 0) * (X[:,0] < -0.5) * (X[:,0] - 0.25)
            return np.column_stack((mu_x1, mu_x2))
        
        if self.s == 4:
            mu_x1 = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
            mu_x2 = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
            # mu_x2 = (X[:,3] * X[:,1] + X[:,0] ** 2 + np.exp(X[:,2] - 1) - 1) * 2
            return np.column_stack((mu_x1, mu_x2))
        
        if self.s == 5:
            mu_x1 = (X[:,0] * X[:,1] > 0) * (X[:,3] * (X[:,3] > 0.5) + 0.5 * (X[:,3] <= 0.5)) + (X[:,0] * X[:,1] <= 0) * (X[:,3] * (X[:,3] < -0.5) - 0.5 * (X[:,3] > -0.5))
            mu_x2 = (X[:,1] * X[:,2] > 0) * (X[:,0] * (X[:,0] > 0.5) + 0.5 * (X[:,0] <= 0.5)) + (X[:,1] * X[:,2] <= 0) * (X[:,0] * (X[:,0] < -0.5) - 0.5 * (X[:,0] > -0.5))
            # mu_x2 = (X[:,0] * X[:,1] > 0) * (X[:,3] * (X[:,3] > 0.5) + 0.5 * (X[:,3] <= 0.5)) + (X[:,0] * X[:,1] <= 0) * (X[:,3] * (X[:,3] < -0.5) - 0.5 * (X[:,3] > -0.5))
            return np.column_stack((mu_x1, mu_x2))
        
        if self.s == 6:
            mu_x1 = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
            mu_x2 = (X[:,1] * X[:,2] + np.exp(X[:,0] - 1)) * 5
            return np.column_stack((mu_x1, mu_x2))
        
        if self.s == 7:
            mu_x1 = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
            mu_x2 = (X[:,2] * X[:,1] > 0) * (X[:,0] > 0.5) * (0.25 + X[:,0]) + (X[:,2] * X[:,1] <= 0) * (X[:,0] < -0.5) * (X[:,0] - 0.25)
            return np.column_stack((mu_x1, mu_x2))
        
        if self.s == 8:
            mu_x1 = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
            mu_x2 = (X[:,3] * X[:,1] + X[:,0] ** 2 + np.exp(X[:,2] - 1) - 1) * 2
            return np.column_stack((mu_x1, mu_x2))
        
class OracleRegressorkd:
    def __init__(self, setting: int):
        assert 1 <= setting and setting <= 2
        self.s = setting
        self.model = LinearRegression()

    def extract(self, X):
        N, d = X.shape
        if self.s == 1:
            U1 = np.zeros((N, d))
            U2 = np.zeros((N, d))
            U3 = np.zeros((N, d))
            for i in range(d):
                U1[:,i] = X[:,(i) % d] * X[:,(i+1) % d]
                U2[:,i] = X[:,(i+2) % d] ** 2 
                U3[:,i] = np.exp(X[:,(i+3) % d] - 1)

            return np.column_stack((U1, U2, U3))
        if self.s == 2:
            # already linear
            return X
        
    def fit(self, X, Y):
        self.model.fit(self.extract(X), Y)

    def predict(self, X):
        return self.model.predict(self.extract(X))

''' 
Wrapper classes for summarizing the best hyperparameters.
'''
class RfRegressor():
    def __init__(self, setting: int, n_estimators=50, max_depth=20, max_features=10, max_leaf_nodes=None):
        assert 1 <= setting <= 8
        if setting != 5:
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=0, max_leaf_nodes=max_leaf_nodes)
        else:
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_leaf=20, min_samples_split=20, random_state=0, max_leaf_nodes=max_leaf_nodes)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class RfRegressor2d():
    def __init__(self, setting: int, n_estimators=50, max_depth=20, max_features=10):
        assert 1 <= setting <= 4
        if setting == 1:
            self.model = RandomForestRegressor(n_estimators=50, max_depth=20, max_features=10, min_samples_leaf=13, min_samples_split=13, random_state=0)
        if setting in [2, 3, 4]:
            self.model = RandomForestRegressor(n_estimators=50, max_depth=20, max_features=10, random_state=0)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
class MlpRegressor():
    def __init__(self, setting: int, hidden_layers=(32, ) * 4):
        assert 1 <= setting <= 8
        if setting == 1:
            self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, random_state=0, alpha=0.5, max_iter=1000, learning_rate_init=1e-4, early_stopping=True, tol=1e-6)
        if setting in [2, 3, 4]:
            self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, random_state=0, alpha=0.5, max_iter=1000, learning_rate_init=1e-4, early_stopping=True, tol=1e-6)
        if setting == 5:
            self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, random_state=0, alpha=2, max_iter=1000, learning_rate_init=1e-4, early_stopping=True, tol=1e-6)
        if setting in [6, 7, 8]:
            self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, random_state=0, alpha=0.2, max_iter=1000, learning_rate_init=1e-4, early_stopping=True, tol=1e-6)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
class MlpRegressor2d():
    def __init__(self, setting: int, hidden_layers=(32, ) * 4):
        assert 1 <= setting <= 4
        if setting in [1, 2, 3, 4]:
            self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, random_state=0, alpha=0.1, max_iter=1000, learning_rate_init=1e-4, early_stopping=True, tol=1e-6)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)