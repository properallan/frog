
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MeanCentering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_ = None
    
    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        return self
    
    def transform(self, X, y=None):
        return X - self.mean_
    
    def inverse_transform(self, X, y=None):
        return X + self.mean_


class SliceMeanCentering(BaseEstimator, TransformerMixin):
    def __init__(self, slices_index=None):
        """
        slices_index: dict com nome do campo -> índices das colunas (ex: {'pressure': slice(0,100)})
        """
        self.slices_index = slices_index
        self.means = {}

    def fit(self, X, y=None):
        from frog.datahandler._array import IndexedArray
        if self.slices_index is None:
            if isinstance(X, IndexedArray):
                self.slices_index = X.index

        for slice_name, slice_index in self.slices_index.items():
            x_slice = X[:, slice_index]
            self.means[slice_name] = x_slice.mean()
        return self

    def transform(self, X):
        X_centered = np.zeros_like(X)
        for slice_name, slice_index in self.slices_index.items():
            x_slice = X[:, slice_index]
            mean = self.means[slice_name]
            X_centered[:, slice_index] = x_slice - mean
        return X_centered

    def inverse_transform(self, X_centered):
        X_rec = np.zeros_like(X_centered)
        for slice_name, slice_index in self.slices_index.items():
            mean = self.means[slice_name]
            X_rec[:, slice_index] = X_centered[:, slice_index] + mean
        return X_rec
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)