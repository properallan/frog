import pickle
from typing import Union
from pathlib import Path
import numpy as np
from frog.datahandler import Indexer
from frog.doe import dict_to_array_and_index, array_and_index_to_dict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared
from frog.transformers import IdentityTransformer
from sklearn.linear_model import LinearRegression
from frog.neuralnetwork import NeuralNetwork
from frog.datahandler import IndexedArray

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

def load(path):
    with open(path, 'rb') as f:
        fr = pickle.load(f)
    return fr

class FR:
    def __init__(self, X_rom=None, y_rom=None, surrogate=None, surrogate_kwargs : dict = {}, load_path : str = None, **kwargs):
        self.X_rom = X_rom
        self.y_rom = y_rom
        self.surrogate = surrogate
        self.surrogate_kwargs = surrogate_kwargs

        if X_rom is None and y_rom is None and surrogate is None and surrogate_kwargs == {}:
            self.load(load_path)
        
        self.setattr(**kwargs)

    def fit(self, X, y, **kwargs):
        fit_kwargs = {**self.surrogate_kwargs, **kwargs}
        print('Performing ROM fit on X data')
        X = self.X_rom.fit_transform(X)
        print('Performing ROM fit on y data')
        y = self.y_rom.fit_transform(y)

        if 'regressor__validation_data' in fit_kwargs.keys():
            if fit_kwargs['regressor__validation_data'] is not None:
                X_validation = fit_kwargs['regressor__validation_data'][0]
                y_validation = fit_kwargs['regressor__validation_data'][1]
                print('Performing ROM fit on X validation data')
                X_validation = self.X_rom.transform(X_validation)
                print('Performing ROM fit on y validation data')
                y_validation = self.y_rom.transform(y_validation)
                fit_kwargs['regressor__validation_data'] = (X_validation, y_validation)
                
        print('Performing surrogate model fit')
        self.surrogate.fit(
            X, 
            y, 
            **fit_kwargs
        )

        return self

class FlowReconstruction(BaseEstimator, TransformerMixin):
    def __init__(self, X_rom=None, y_rom=None, surrogate=None, surrogate_kwargs : dict = {}, **kwargs):
        self.X_rom = X_rom
        self.y_rom = y_rom
        self.surrogate = surrogate
        self.surrogate_kwargs = surrogate_kwargs
        self.y_index = None

    def fit(self, X, y, **kwargs):
        if isinstance(y, IndexedArray):
            self.y_index = y.index
            
        fit_kwargs = {**self.surrogate_kwargs, **kwargs}
        print('Performing ROM fit on X data')
        X = self.X_rom.fit_transform(X)
        print('Performing ROM fit on y data')
        y = self.y_rom.fit_transform(y)

        if 'regressor__validation_data' in fit_kwargs.keys():
            if fit_kwargs['regressor__validation_data'] is not None:
                X_validation = fit_kwargs['regressor__validation_data'][0]
                y_validation = fit_kwargs['regressor__validation_data'][1]
                print('Performing ROM fit on X validation data')
                X_validation = self.X_rom.transform(X_validation)
                print('Performing ROM fit on y validation data')
                y_validation = self.y_rom.transform(y_validation)
                fit_kwargs['regressor__validation_data'] = (X_validation, y_validation)
                
        print('Performing surrogate model fit')
        self.surrogate.fit(
            X, 
            y, 
            **fit_kwargs
        )

        return self
    
    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X, y)
    
    def transform(self, X, y=None):
        X_in = self.X_rom.transform(X)
        y_out = self.surrogate.predict(X_in)
        y_out = self.y_rom.inverse_transform(y_out)

        if self.y_index is not None:
            y_out = IndexedArray(
                input_array=y_out, 
                index=self.y_index, 
                doe_index=None,
                doe_file=None)
            
        return y_out

    def inverse_transform(self, y):
        raise NotImplementedError

    def set_dataset(self, 
        training_X_file: Path=None,
        training_y_file: Path=None,
        test_X_file: Path=None,
        test_y_file: Path=None,
        validation_X_file: Path=None,
        validation_y_file: Path=None,
        low_fidelity_variables: list=None,
        high_fidelity_variables: list=None):

        if training_X_file is not None:
            X_train = np.load(training_X_file, allow_pickle=True)
            snapshots_X_train = X_train['snapshots']
            index_X_train = X_train['snapshot_index'].item()    
            X_train_dict = array_and_index_to_dict(snapshots_X_train, index_X_train)
            self.snapshots_X_train, self.idx_dict_X_train = dict_to_array_and_index(X_train_dict, low_fidelity_variables)
        else:
            raise ValueError('training_X_file must be provided')        

        if training_y_file is not None:
            y_train = np.load(training_y_file, allow_pickle=True)
            snapshots_y_train = y_train['snapshots']
            index_y_train = y_train['snapshot_index'].item()
            y_train_dict = array_and_index_to_dict(snapshots_y_train, index_y_train)
            self.snapshots_y_train, self.idx_dict_y_train = dict_to_array_and_index(y_train_dict, high_fidelity_variables)
        else:
            raise ValueError('training_y_file must be provided')

        if test_X_file is not None:
            X_test = np.load(test_X_file, allow_pickle=True)
            snapshots_X_test = X_test['snapshots']
            index_X_test = X_test['snapshot_index'].item()
            X_test_dict = array_and_index_to_dict(snapshots_X_test, index_X_test)
            self.snapshots_X_test, self.idx_dict_X_test = dict_to_array_and_index(X_test_dict, low_fidelity_variables)
        else:
            self.snapshots_X_test = None

        if test_y_file is not None:    
            y_test = np.load(test_y_file, allow_pickle=True)
            snapshots_y_test = y_test['snapshots']
            index_y_test = y_test['snapshot_index'].item()
            y_test_dict = array_and_index_to_dict(snapshots_y_test, index_y_test)
            self.snapshots_y_test, self.idx_dict_y_test = dict_to_array_and_index(y_test_dict, high_fidelity_variables)
        else:
            self.snapshots_y_test = None

        if validation_X_file is not None:
            X_validation = np.load(validation_X_file, allow_pickle=True)
            snapshots_X_validation = X_validation['snapshots']
            index_X_validation = X_validation['snapshot_index'].item()
            X_validation_dict = array_and_index_to_dict(snapshots_X_validation, index_X_validation)
            self.snapshots_X_validation, self.idx_dict_X_validation = dict_to_array_and_index(X_validation_dict, low_fidelity_variables)
        else:
            self.snapshots_X_validation = None

        if validation_y_file is not None:
            y_validation = np.load(validation_y_file, allow_pickle=True)
            snapshots_y_validation = y_validation['snapshots']
            index_y_validation = y_validation['snapshot_index'].item()
            y_validation_dict = array_and_index_to_dict(snapshots_y_validation, index_y_validation)
            self.snapshots_y_validation, self.idx_dict_y_validation = dict_to_array_and_index(y_validation_dict, high_fidelity_variables)
        else:
            self.snapshots_y_validation = None

        self.VALIDATION_DATA = (self.snapshots_X_validation, self.snapshots_y_validation)

    def predict(self, X):
        X_in = self.X_rom.transform(X)
        y_out = self.surrogate.predict(X_in)
        y_out = self.y_rom.inverse_transform(y_out)
        
        return y_out
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

        return self
    
    def load(self, path):
        with open(path, 'rb') as f:
            self = pickle.load(f)
        
        return self
    
    def setattr(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Abstract class
class FRBuilder(FR):
    def __init__(self, X_rom, y_rom, surrogate, surrogate_kwargs, **kwargs) -> None:
        super().__init__(
            X_rom=X_rom, 
            y_rom=y_rom, 
            surrogate=surrogate, 
            surrogate_kwargs=surrogate_kwargs
        )

        self.TRAINING_X = Path(self.TRAINING_X)
        self.TRAINING_y = Path(self.TRAINING_y)
        self.TEST_X = Path(self.TEST_X)
        self.TEST_y = Path(self.TEST_y)
        self.VALIDATION_X = Path(self.VALIDATION_X)
        self.VALIDATION_y = Path(self.VALIDATION_y)
        
        self.set_dataset(
            training_X_file=self.TRAINING_X,
            training_y_file=self.TRAINING_y,
            test_X_file=self.TEST_X,
            test_y_file=self.TEST_y,
            validation_X_file=self.VALIDATION_X,
            validation_y_file=self.VALIDATION_y,
            low_fidelity_variables=self.LF_VARIABLES,
            high_fidelity_variables=self.HF_VARIABLES
        )
       
    def build(self, **kwargs):
        # Not implemented
        pass

    def metrics(self):
        # Not implemented
        pass

    def plot(self, variable):
        # Not implemented
        pass

    def plot_ith_prediction(self, ground_truth, prediction, i, variable):
        import matplotlib.pyplot as plt
        plt.figure()
        idx_dict = self.idx_dict_y_test
        plt.plot(ground_truth[i][idx_dict[variable]])
        plt.plot(prediction[i][idx_dict[variable]], ls='-.')
        
    def plot_predictions(self, variable='UPPER_WALL/Heat_Flux'):
        prediction = self.predict(self.snapshots_X_test)
        ground_truth = self.snapshots_y_test

        for i in range(len(ground_truth)):
            self.plot_ith_prediction(ground_truth, prediction, i, variable)
    

class FRLinearBuilder(FRBuilder):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        X_scaler = ('scaler', MinMaxScaler())
        y_scaler = ('scaler', MinMaxScaler())
        
        if self.N_COMPONENTS_X is None:
            X_reducer = ('reducer', IdentityTransformer())
        else:
            X_reducer = ('reducer', TruncatedSVD(n_components=self.N_COMPONENTS_X))
        
        if self.N_COMPONENTS_y is None:
            y_reducer = ('reducer', IdentityTransformer())
        else:
            y_reducer = ('reducer', TruncatedSVD(n_components=self.N_COMPONENTS_y))

        X_rom = Pipeline([
            X_scaler ,
            X_reducer,
        ])

        y_rom = Pipeline([
            y_scaler,
            y_reducer,
        ])

        surrogate = Pipeline([
            ('regressor', LinearRegression())
        ])
        
        surrogate_kwargs={}

        super().__init__(
            X_rom=X_rom, 
            y_rom=y_rom, 
            surrogate=surrogate, 
            surrogate_kwargs=surrogate_kwargs)

class FRKrigingBuilder(FRBuilder):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        X_scaler = ('scaler', MinMaxScaler())
        y_scaler = ('scaler', MinMaxScaler())

        
        if self.N_COMPONENTS_X is None:
            X_reducer = ('reducer', IdentityTransformer())
        else:
            X_reducer = ('reducer', TruncatedSVD(n_components=self.N_COMPONENTS_X))
        
        if self.N_COMPONENTS_y is None:
            y_reducer = ('reducer', IdentityTransformer())
        else:
            y_reducer = ('reducer', TruncatedSVD(n_components=self.N_COMPONENTS_y))

        X_rom = Pipeline([
            X_scaler ,
            X_reducer,
        ])

        y_rom = Pipeline([
            y_scaler,
            y_reducer,
        ])

        kernel = 1.0 * ExpSineSquared(
            length_scale=1.0,
            periodicity=3.0,
            length_scale_bounds=(0.1, 10.0),
            periodicity_bounds=(1.0, 10.0),
        )

        kernel = 1.0 * RBF(
            length_scale=1.0, 
            length_scale_bounds=(1e-1, 10.0)
        )

        surrogate = Pipeline([
            ('regressor', GaussianProcessRegressor()
                #kernel=kernel,
                #alpha=1e-10,
                #normalize_y=True,
                #n_restarts_optimizer=1,
                #optimizer='fmin_l_bfgs_b',
                #)
            )
        ])
        
        surrogate_kwargs={}

        super().__init__(
            X_rom=X_rom, 
            y_rom=y_rom, 
            surrogate=surrogate, 
            surrogate_kwargs=surrogate_kwargs)

class FRNNBuilder(FRBuilder):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        X_scaler = ('scaler', MinMaxScaler())
        y_scaler = ('scaler', MinMaxScaler())
        
        if self.N_COMPONENTS_X is None:
            X_reducer = ('reducer', IdentityTransformer())
        else:
            X_reducer = ('reducer', TruncatedSVD(n_components=self.N_COMPONENTS_X))
        
        if self.N_COMPONENTS_y is None:
            y_reducer = ('reducer', IdentityTransformer())
        else:
            y_reducer = ('reducer', TruncatedSVD(n_components=self.N_COMPONENTS_y))

        X_rom = Pipeline([
            X_scaler ,
            X_reducer,
        ])

        y_rom = Pipeline([
            y_scaler,
            y_reducer,
        ])

        surrogate = Pipeline([ 
                    ('regressor', NeuralNetwork(    
                    num_inputs=self.N_COMPONENTS_X,
                    num_outputs=self.N_COMPONENTS_y,
                    num_layers=self.N_LAYERS,
                    num_neurons=self.N_NEURONS,
                    activation=self.ACTIVATION,
                    optimizer=self.OPTIMIZER, 
                    loss=self.LOSS,)),
        ])[0]

        surrogate_kwargs=dict( 
            epochs= self.EPOCHS, 
            batch_size= self.BATCH_SIZE, 
        )

        super().__init__(X_rom=X_rom, y_rom=y_rom, surrogate=surrogate, surrogate_kwargs=surrogate_kwargs)
