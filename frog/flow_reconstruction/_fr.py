import dill
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
    
    def transform(self, X, y=None, **kwargs):
        X_in = self.X_rom.transform(X)
        y_out = self.surrogate.predict(X_in, **kwargs)

        if 'return_std' in kwargs.keys():
            from copy import copy

            out = copy(y_out)
            y_out = out[0]
            std = out[1]
        
            y_out = self.y_rom.inverse_transform(y_out)
            std = self.y_rom.inverse_transform(std)
        else:
            y_out = self.y_rom.inverse_transform(y_out)

        if self.y_index is not None:
            y_out = IndexedArray(
                input_array=y_out, 
                index=self.y_index, 
                doe_index=None,
                doe_file=None)
            if 'return_std' in kwargs.keys():
                std = IndexedArray(
                    input_array=std, 
                    index=self.y_index, 
                    doe_index=None,
                    doe_file=None)
            
        if 'return_std' in kwargs.keys():
            y_out = (y_out, std)
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

    def predict(self, X, **kwargs):
        y_out = self.transform(X, **kwargs)

        return y_out
    
    # def save(self, path):
    #     """
    #     Salva o FlowReconstruction separadamente do modelo Keras.
    #     """
    #     from pathlib import Path

    #     if hasattr(self.surrogate.named_steps['regressor'], 'model'):  
    #         # Salvar modelo keras
    #         model_path = Path(path) / "fr_model_surrogate_model.h5"
    #         self.surrogate.named_steps['regressor'].model.save(model_path)

    #         # Antes de salvar o objeto, remover o keras model da surrogate
    #         model_backup = self.surrogate.named_steps['regressor'].model
    #         self.surrogate.named_steps['regressor'].model = None

    #         # Salvar o objeto via pickle
    #         with open(Path(path) / "fr_model_flow.pkl", 'wb') as f:
    #             dill.dump(self, f)

    #         # Restaurar o modelo em memória
    #         self.surrogate.named_steps['regressor'].model = model_backup

    #         print(f"FlowReconstruction salvo em {path}_flow.pkl e modelo salvo em {path}_surrogate_model.h5")

    #     else:
    #         if not Path(path).exists():
    #             Path(path).mkdir(parents=True, exist_ok=True)
    #         # Salvar o objeto via pickle
    #         with open(Path(path) / "fr_model_flow.pkl", 'wb') as f:
    #             dill.dump(self, f)

    #         print(f"FlowReconstruction salvo em {path}_flow.pkl")
    #     return self

    def save(self, path):
        """
        Save the FlowReconstruction object and its components safely.
        """
        from pathlib import Path
        import dill
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        keras_model = None
        has_keras_model = (
            hasattr(self.surrogate, 'named_steps') and
            'regressor' in self.surrogate.named_steps and
            hasattr(self.surrogate.named_steps['regressor'], 'model') and
            self.surrogate.named_steps['regressor'].model is not None
        )

        if has_keras_model:
            keras_model = self.surrogate.named_steps['regressor'].model
            model_path = path / "fr_model_surrogate_model.h5"
            keras_model.save(model_path)
            self.surrogate.named_steps['regressor'].model = None

        # Salvar componentes separadamente
        joblib.dump(self.X_rom, path / "X_rom.pkl")
        joblib.dump(self.y_rom, path / "y_rom.pkl")
        #joblib.dump(self.surrogate, path / "surrogate.pkl")
        with open(path / "surrogate.pkl", 'wb') as f:
            dill.dump(self.surrogate, f)

        # Salvar o restante da classe (sem os componentes já salvos)
        tmp_X_rom = self.X_rom
        tmp_y_rom = self.y_rom
        tmp_surrogate = self.surrogate

        self.X_rom = None
        self.y_rom = None
        self.surrogate = None

        with open(path / "fr_model_flow.pkl", 'wb') as f:
            dill.dump(self, f)

        # Restaurar objetos em memória
        self.X_rom = tmp_X_rom
        self.y_rom = tmp_y_rom
        self.surrogate = tmp_surrogate
        if has_keras_model:
            self.surrogate.named_steps['regressor'].model = keras_model

        print(f"Saved model to: {path}")
    
    @staticmethod
    def load(path):
        """
        Load the FlowReconstruction object and reattach its components.
        """
        from pathlib import Path
        import dill
        import joblib
        from tensorflow import keras

        path = Path(path)

        # Carrega o objeto principal
        with open(path / "fr_model_flow.pkl", 'rb') as f:
            obj = dill.load(f)

        # Carrega os componentes salvos separadamente
        obj.X_rom = joblib.load(path / "X_rom.pkl")
        obj.y_rom = joblib.load(path / "y_rom.pkl")
        #obj.surrogate = dill.load(path / "surrogate.pkl")
        with open(path / "surrogate.pkl", "rb") as f:
            obj.surrogate = dill.load(f)
        

        # Restaura o modelo Keras se existir
        model_path = path / "fr_model_surrogate_model.h5"
        if model_path.exists():
            if (
                hasattr(obj.surrogate, 'named_steps') and
                'regressor' in obj.surrogate.named_steps and
                hasattr(obj.surrogate.named_steps['regressor'], 'model')
            ):
                obj.surrogate.named_steps['regressor'].model = keras.models.load_model(model_path, compile=False)
                print(f"Loaded model and Keras model from: {path}")
            else:
                print("Keras model found but regressor has no `model` attribute.")
        else:
            print(f"Loaded model (no Keras model) from: {path}")

        return obj


    # @staticmethod
    # def load(path):
    #     """
    #     Carrega o FlowReconstruction e seu modelo Keras.
    #     """
    #     from pathlib import Path
    #     from tensorflow import keras
    #     import os
    #     # Carregar o objeto
    #     with open(Path(path) / "fr_model_flow.pkl", 'rb') as f:
    #         obj = dill.load(f)

    #     if os.path.exists(Path(path) / "fr_model_surrogate_model.h5"):
    #         # Carregar o modelo keras
    #         model_path = Path(path) / "fr_model_surrogate_model.h5"
    #         obj.surrogate.named_steps['regressor'].model = keras.models.load_model(model_path, compile=False)

    #         print(f"FlowReconstruction carregado de {path}/fr_model_flow.pkl e modelo de {path}/fr_model_surrogate_model.h5")

    #     else:
    #         print(f"FlowReconstruction carregado de {path}/fr_model_flow.pkl.")
    #     return obj
    
    def setattr(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

