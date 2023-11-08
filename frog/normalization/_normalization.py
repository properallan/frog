from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class DataHandlerMinMaxScaler(MinMaxScaler):
    def __init__(self, datahandler, feature_range=(0,1)):
        self.datahandler = datahandler
        self.scalers = {}
        self.feature_range = feature_range

        for dataset in self.datahandler.datasets:
            self.scalers[dataset] = MinMaxScaler(feature_range=self.feature_range)

    def fit(self, X):
        print('fit')
        for dataset in self.datahandler.datasets:
            self.scalers[dataset].fit(self.datahandler(X)[dataset])

    def transform(self, X):
        print('transform')
        Xt = X.copy()
        for dataset in self.datahandler.datasets:
            Xt[self.datahandler.get_index(dataset)] = self.scalers[dataset].transform(X[self.datahandler.get_index(dataset)])
        return Xt
    
    def inverse_transform(self, X):
        print('inverse_transform')
        Xt = X.copy()
        for dataset in self.datahandler.datasets:
            Xt[self.datahandler.get_index(dataset)] = self.scalers[dataset].inverse_transform(X[self.datahandler.get_index(dataset)])
        return Xt
    
    def fit_transform(self, X, y=None):
        print('fit_transform')
        self.fit(X)
        return self.transform(X)

class DataHandlerStandardNormalization(BaseEstimator, TransformerMixin):
    def __init__(self,datahandler, bounds):
        self.bounds = bounds
        self.datahandler = datahandler

    def fit(self, X, y=None):
        print('fit')
        self.mean = {}
        self.min = {}
        self.max = {}
        self.std = {}
        for dataset in self.datahandler.datasets:
            self.mean[dataset] = np.mean(self.datahandler(X)[dataset], axis=0)
            self.std[dataset] = np.std(self.datahandler(X)[dataset], axis=0)
            self.min[dataset] = np.min(self.datahandler(X)[dataset], axis=0)
            self.max[dataset] = np.max(self.datahandler(X)[dataset], axis=0)
    
    def transform(self, X, y=None):
        print('transform')
        for dataset in self.datahandler.datasets:
            X[self.datahandler.get_index(dataset)] -=  self.mean[dataset]
            X[self.datahandler.get_index(dataset)] = (self.bounds[1]-self.bounds[0]) * (X[self.datahandler.get_index(dataset)]) / (self.std[dataset]) + self.bounds[0]
        return X

    def inverse_transform(self, X, y=None):
        print('inverse_transform')
        for dataset in self.datahandler.datasets:
            X[self.datahandler.get_index(dataset)] = (self.std[dataset]) * (X[self.datahandler.get_index(dataset)] - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
            X[self.datahandler.get_index(dataset)] +=  self.mean[dataset]
        return X

    def fit_transform(self, X, y=None):
        print('fit_transform')

        self.fit(X)
        return self.transform(X)

class DataHandlerNormalization(BaseEstimator, TransformerMixin):
    def __init__(self,datahandler, bounds):
        self.bounds = bounds
        self.datahandler = datahandler

    def fit(self, X, y=None):
        print('fit')
        self.mean = {}
        self.min = {}
        self.max = {}
        for dataset in self.datahandler.datasets:
            self.mean[dataset] = np.mean(self.datahandler(X)[dataset], axis=0)
            self.min[dataset] = np.min(self.datahandler(X)[dataset])
            self.max[dataset] = np.max(self.datahandler(X)[dataset])
    
    def transform(self, X, y=None):
        print('transform')
        for dataset in self.datahandler.datasets:
            X[self.datahandler.get_index(dataset)] -=  self.mean[dataset]
            X[self.datahandler.get_index(dataset)] = (self.bounds[1]-self.bounds[0]) * (X[self.datahandler.get_index(dataset)] - self.min[dataset]) / (self.max[dataset] - self.min[dataset]) + self.bounds[0]
        return X

    def inverse_transform(self, X, y=None):
        print('inverse_transform')
        for dataset in self.datahandler.datasets:
            X[self.datahandler.get_index(dataset)] = (self.max[dataset] - self.min[dataset]) * (X[self.datahandler.get_index(dataset)] - self.bounds[0]) / (self.bounds[1] - self.bounds[0]) + self.min[dataset]
            X[self.datahandler.get_index(dataset)] +=  self.mean[dataset]
        return X

    def fit_transform(self, X, y=None):
        print('fit_transform')

        self.fit(X)
        return self.transform(X)
    
class DataHandlerNormalization2(BaseEstimator, TransformerMixin):
    def __init__(self,datahandler, bounds):
        self.bounds = bounds
        self.datahandler = datahandler

    def fit(self, X, y=None):
        print('fit')
        self.mean = {}
        self.min = {}
        self.max = {}
        for dataset in self.datahandler.datasets:
            self.mean[dataset] = np.mean(self.datahandler(X)[dataset], axis=0)
            self.min[dataset] = np.min(self.datahandler(X)[dataset])
            self.max[dataset] = np.max(self.datahandler(X)[dataset])
    
    def transform(self, X, y=None):
        print('transform')
        for dataset in self.datahandler.datasets:
            X[self.datahandler.get_index(dataset)] -=  self.mean[dataset]
            X[self.datahandler.get_index(dataset)] = (self.bounds[1]-self.bounds[0]) * (X[self.datahandler.get_index(dataset)] - self.min[dataset]) / (self.max[dataset] - self.min[dataset]) + self.bounds[0]
        return X

    def inverse_transform(self, X, y=None):
        print('inverse_transform')
        for dataset in self.datahandler.datasets:
            X[self.datahandler.get_index(dataset)] = (self.max[dataset] - self.min[dataset]) * (X[self.datahandler.get_index(dataset)] - self.bounds[0]) / (self.bounds[1] - self.bounds[0]) + self.min[dataset]
            X[self.datahandler.get_index(dataset)] +=  self.mean[dataset]
        return X

    def fit_transform(self, X, y=None):
        print('fit_transform')

        self.fit(X)
        return self.transform(X)


class Normalization(BaseEstimator, TransformerMixin):
    def __init__(self, bounds=[-1,1]):
        self.bounds = bounds

    def fit(self, X, y=None):
        
        self.mean = np.mean(X, axis=0)
        self.min = np.min(X)
        self.max = np.max(X)

    def fit_transform(self, X, y=None):
        self.fit(X)

        return self.transform(X)

    def transform(self, X, y=None):
        X = X - self.mean
        X = (self.bounds[1]-self.bounds[0]) * (X - self.min) / (self.max - self.min) + self.bounds[0]
        #X = np.nan_to_num(X, 0)
        return X

    def inverse_transform(self, X, y=None):
        print('inverse_transform')
        X = (self.max - self.min) * (X - self.bounds[0]) / (self.bounds[1] - self.bounds[0]) + self.min
        X = X + self.mean
        return X