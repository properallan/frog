
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MeanCenteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_ = None
    
    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        return self
    
    def transform(self, X, y=None):
        return X - self.mean_
    
    def inverse_transform(self, X, y=None):
        return X + self.mean_
