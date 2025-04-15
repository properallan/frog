from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from frog.datahandler._array import IndexedArray

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class SliceMaxAbsScaler(BaseEstimator, TransformerMixin):
    def __init__(self, slices_index=None,):
        """
        slices_index: dict com nome do campo -> índices das colunas (ex: {'pressure': slice(0,100)})
        """
        self.slices_index = slices_index
        self.feature_range = {}
        self.denom = {}
        self.X_min = {}
        self.X_max = {}

    def fit(self, X, y=None):
        if self.slices_index is None:
            if isinstance(X, IndexedArray):
                self.slices_index = X.index

        for slice_name, slice_index in self.slices_index.items():
            slice = X[:, slice_index]
            slice_min = slice.min()
            slice_max = slice.max()
            if slice_min >= 0:
                # Todos os valores positivos → escalar para [0, 1]
                self.denom[slice_name] = np.abs(slice_max)
            else:
                # Existem negativos → escalar para [-1, 1]
                self.denom[slice_name] = np.maximum(np.abs(slice_min), np.abs(slice_max))
                if self.denom[slice_name] == 0.0:
                    self.denom[slice_name] = 1.0
            
        return self

    def transform(self, X):
        X_scaled = np.zeros_like(X)
        for slice_name, slice_index in self.slices_index.items():
            slice = X[:, slice_index]
            X_scaled[:, slice_index] = slice / self.denom[slice_name]
        return X_scaled
    

    def inverse_transform(self, X_scaled):
        X_rec = np.zeros_like(X_scaled)
        for slice_name, slice_index in self.slices_index.items():
            X_rec[:, slice_index] = X_scaled[:, slice_index] * self.denom[slice_name]
        return X_rec
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class SliceMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, slices_index=None, feature_range=(0, 1)):
        """
        slices_index: dict com nome do campo -> índices das colunas (ex: {'pressure': slice(0,100)})
        feature_range: intervalo desejado após normalização (ex: (0,1) ou (-1,1))
        """
        self.slices_index = slices_index
        self.feature_range = feature_range
        self.X_min = {}
        self.X_max = {}

    def fit(self, X, y=None):
        if self.slices_index is None:
            if isinstance(X, IndexedArray):
                self.slices_index = X.index

        for slice_name, slice_index in self.slices_index.items():
            X_slice = X[:, slice_index]
            self.X_min[slice_name] = X_slice.min(axis=0)
            self.X_max[slice_name] = X_slice.max(axis=0)
        return self

    def transform(self, X):
        X_scaled = np.zeros_like(X)
        min_val, max_val = self.feature_range
        scale = max_val - min_val

        for slice_name, slice_index in self.slices_index.items():
            X_slice = X[:, slice_index]
            X_min = self.X_min[slice_name]
            X_max = self.X_max[slice_name]
            denom = X_max - X_min
            denom[denom == 0] = 1.0  # Evita divisão por zero

            X_scaled[:, slice_index] = ((X_slice - X_min) / denom) * scale + min_val

        return X_scaled

    def inverse_transform(self, X_scaled):
        X_rec = np.zeros_like(X_scaled)
        min_val, max_val = self.feature_range
        scale = max_val - min_val

        for slice_name, slice_index in self.slices_index.items():
            X_slice = X_scaled[:, slice_index]
            X_min = self.X_min[slice_name]
            X_max = self.X_max[slice_name]
            denom = X_max - X_min
            denom[denom == 0] = 1.0

            X_rec[:, slice_index] = ((X_slice - min_val) / scale) * denom + X_min

        return X_rec

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)




class PhysicalNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, slices_index=None, slices_ref=None):
        """
        refs: lista com os valores de referência para cada campo.
              Use None para campos que não devem ser normalizados (ex: Mach).
              Exemplo: [p_ref, None, T_ref, q_ref]
        """
        self.slices_ref = slices_ref
        self.slices_index = slices_index
        self.mean = {}

    def fit(self, X, y=None):
        from frog.datahandler._array import IndexedArray
        
        if self.slices_index is None:
            if isinstance(X, IndexedArray):
                self.slices_index = X.index

        if self.slices_ref is None and isinstance(X, IndexedArray):
            self.slices_ref = {}
            for slice_name, slice_index in self.slices_index.items():
                self.slices_ref[slice_name] = np.maximum(np.max(X[:, slice_index]),  np.min(X[:, slice_index]))
        
        if self.slices_ref is not None and self.slices_index is not None:
            for slice_name, slice_index in self.slices_index.items():
                if slice_name not in self.slices_ref.keys():
                    self.slices_ref[slice_name] = np.maximum(np.max(X[:, slice_index]),  np.min(X[:, slice_index]))
                self.mean[slice_name] = np.mean(X[:, slice_index])
        
        return self

    def transform(self, X):
        X_norm = np.zeros_like(X)
        for slice_name, slice_index in self.slices_index.items():
            ref = self.slices_ref[slice_name]
            X_norm[:, slice_index] = (X[:, slice_index] - self.mean[slice_name]) / ref
        return X_norm
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_norm):
        X_rec = np.zeros_like(X_norm)
        for slice_name, slice_index in self.slices_index.items():
            ref = self.slices_ref[slice_name]
            X_rec[:, slice_index] = X_norm[:, slice_index] * ref + self.mean[slice_name]
        return X_rec

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
    
class RowScalerRaw(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.axis=1

    def fit(self, X, y=None):
        axis = self.axis
        self.min = 0
        self.max = 1

        self.X_min = X.min(axis=axis).min()
        self.X_max = X.max(axis=axis).max()

    def transform(self, X, y=None):
        self.X_std = (X.T - self.X_min) / (self.X_max - self.X_min)
        self.X_std = self.X_std.T
        self.X_scaled = self.X_std * (self.max - self.min) + self.min
        return self.X_scaled   
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X, y=None):
        self.X_std = (X.T - self.min) / (self.max - self.min)
        self.X_std = self.X_std.T
        self.X_descaled = self.X_std * (self.X_max - self.X_min) + self.X_min
        return self.X_descaled
    
class RowScaler(BaseEstimator, TransformerMixin):
    def __init__(self, idx_dict={}):
        self.axis=1
        self.idx_dict = idx_dict

    def fit(self, X, y=None):
        axis = self.axis
        self.min = 0
        self.max = 1

        self.X_min = {}
        self.X_max = {}
        for key, value in self.idx_dict.items():
            # print(X[:,value])
            # print(X[:,value].min(axis=axis).min())
            # input()
            self.X_min[key] = X[:,value].min(axis=axis).min()
            self.X_max[key] = X[:,value].max(axis=axis).max()
        #self.X_min = X.min(axis=axis).min()
        #self.X_max = X.max(axis=axis).max()

    def transform(self, X, y=None):
        self.X_scaled = {}
        self.X_std = {}
        for key, value in self.idx_dict.items():
            self.X_std[key] = (X[:,value].T - self.X_min[key]) / (self.X_max[key] - self.X_min[key])
            self.X_std[key] = self.X_std[key].T
            self.X_scaled[key] = self.X_std[key] * (self.max - self.min) + self.min

        self.X_scaled_array = np.zeros_like(X)
        for key, value in self.X_scaled.items():
            self.X_scaled_array[:,self.idx_dict[key]] = value

        return self.X_scaled_array
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        self.X_descaled = {}
        for key, value in self.idx_dict.items():
            self.X_std[key] = (X[:,value].T - self.min) / (self.max - self.min)
            self.X_std[key] = self.X_std[key].T
            self.X_descaled[key] = self.X_std[key] * (self.X_max[key] - self.X_min[key]) + self.X_min[key]

        self.X_descaled_array = np.zeros_like(X)
        for key, value in self.X_descaled.items():
            self.X_descaled_array[:,self.idx_dict[key]] = value

        return self.X_descaled_array
    
class RowScalerNpz(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, npz_file: str=None):
        self.axis=1
        self.npz_file = npz_file
        self.idx_dict = np.load(self.npz_file, allow_pickle=True)['snapshot_index'].item()

    def fit(self, X, y=None):
        axis = self.axis
        self.min = 0
        self.max = 1

        self.X_min = {}
        self.X_max = {}
        for key, value in self.idx_dict.items():
            # print(X[:,value])
            # print(X[:,value].min(axis=axis).min())
            # input()
            self.X_min[key] = X[:,value].min(axis=axis).min()
            self.X_max[key] = X[:,value].max(axis=axis).max()
        #self.X_min = X.min(axis=axis).min()
        #self.X_max = X.max(axis=axis).max()

    def transform(self, X, y=None):
        self.X_scaled = {}
        self.X_std = {}
        for key, value in self.idx_dict.items():
            self.X_std[key] = (X[:,value].T - self.X_min[key]) / (self.X_max[key] - self.X_min[key])
            self.X_std[key] = self.X_std[key].T
            self.X_scaled[key] = self.X_std[key] * (self.max - self.min) + self.min

        self.X_scaled_array = np.zeros_like(X)
        for key, value in self.X_scaled.items():
            self.X_scaled_array[:,self.idx_dict[key]] = value

        return self.X_scaled_array
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        self.X_descaled = {}
        for key, value in self.idx_dict.items():
            self.X_std[key] = (X[:,value].T - self.min) / (self.max - self.min)
            self.X_std[key] = self.X_std[key].T
            self.X_descaled[key] = self.X_std[key] * (self.X_max[key] - self.X_min[key]) + self.X_min[key]

        self.X_descaled_array = np.zeros_like(X)
        for key, value in self.X_descaled.items():
            self.X_descaled_array[:,self.idx_dict[key]] = value

        return self.X_descaled_array
    
class RowScalerIndexedArray(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    def __init__(self, idx_dict=None, axis=1):
        self.axis=axis
        self.idx_dict = idx_dict

    def fit(self, X, y=None):
        if self.idx_dict is None:
            self.idx_dict = X.index
        
        axis = self.axis
        self.min = 0
        self.max = 1

        self.X_min = {}
        self.X_max = {}
        for key, value in self.idx_dict.items():
            self.X_min[key] = X[:,value].min(axis=axis).min()
            self.X_max[key] = X[:,value].max(axis=axis).max()

    def transform(self, X, y=None):
        self.X_scaled = {}
        self.X_std = {}
        for key, value in self.idx_dict.items():
            self.X_std[key] = (X[:,value].T - self.X_min[key]) / (self.X_max[key] - self.X_min[key])
            self.X_std[key] = self.X_std[key].T
            self.X_scaled[key] = self.X_std[key] * (self.max - self.min) + self.min

        X_scaled_array = np.asarray(np.zeros_like(X))
        for key, value in self.X_scaled.items():
            X_scaled_array[:,self.idx_dict[key]] = value

        if isinstance(X, IndexedArray):
            X_scaled_array = IndexedArray(X_scaled_array, X.index, X.doe_index, X.doe_file)
        self.X_scaled_array = X_scaled_array

        return self.X_scaled_array
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        self.X_descaled = {}
        for key, value in self.idx_dict.items():
            self.X_std[key] = (X[:,value].T - self.min) / (self.max - self.min)
            self.X_std[key] = self.X_std[key].T
            self.X_descaled[key] = self.X_std[key] * (self.X_max[key] - self.X_min[key]) + self.X_min[key]

        X_descaled_array = np.asarray(np.zeros_like(X))
        for key, value in self.X_descaled.items():
            X_descaled_array[:,self.idx_dict[key]] = value

        if isinstance(X, IndexedArray):
            X_descaled_array = IndexedArray(X_descaled_array, X.index, X.doe_index, X.doe_file)
        self.X_descaled_array = X_descaled_array

        return self.X_descaled_array