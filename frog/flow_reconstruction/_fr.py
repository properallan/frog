import dill
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
        return dill.load(f)  # em vez de pickle.load

def _find_keras_model(holder):
    """
    Tenta achar um modelo Keras dentro de um objeto que pode ser
    um Pipeline, um estimador simples, etc.
    Retorna (obj_ref, attr_name) se achar (para setar None/restaurar),
    e o próprio model. Caso contrário, retorna (None, None, None).
    """
    try:
        # Caso Pipeline com passo 'regressor'
        if hasattr(holder, "named_steps") and 'regressor' in holder.named_steps:
            reg = holder.named_steps['regressor']
            if hasattr(reg, 'model') and reg.model is not None:
                return (reg, 'model', reg.model)
        # Caso o próprio holder tenha .model
        if hasattr(holder, 'model') and holder.model is not None:
            return (holder, 'model', holder.model)
    except Exception:
        pass
    return (None, None, None)

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
        from copy import copy

        X_in = self.X_rom.transform(X)
        y_out = self.surrogate.predict(X_in, **kwargs)

        self.X_latent = copy(X_in)
        self.y_latent = copy(y_out)

        if 'return_std' in kwargs.keys():
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
    
    def save(self, path):
        from pathlib import Path
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        keras_owner, keras_attr, keras_model = _find_keras_model(self.surrogate)

        # 1) Se houver modelo Keras, salvar separadamente e retirar antes do dill
        model_path = path / "fr_model_surrogate_model.h5"
        if keras_model is not None:
            keras_model.save(model_path)
            # backup e remoção
            model_backup = keras_model
            setattr(keras_owner, keras_attr, None)
        else:
            model_backup = None

        # 2) Salvar o objeto principal com dill
        pkl_path = path / "fr_model_flow.pkl"
        with open(pkl_path, 'wb') as f:
            dill.dump(self, f)

        # 3) Restaurar o ponteiro em memória (não afeta o arquivo)
        if model_backup is not None:
            setattr(keras_owner, keras_attr, model_backup)

        print(f"FlowReconstruction salvo em {pkl_path}")
        if keras_model is not None:
            print(f"Modelo Keras salvo em {model_path}")
        return self

    @staticmethod
    def load(path):
        from pathlib import Path
        from tensorflow import keras

        path = Path(path)
        pkl_path = path / "fr_model_flow.pkl"
        model_path = path / "fr_model_surrogate_model.h5"

        # 1) Carrega o objeto dill
        with open(pkl_path, 'rb') as f:
            obj = dill.load(f)

        # 2) Se existir um arquivo Keras, restaura
        if model_path.exists():
            keras_owner, keras_attr, _ = _find_keras_model(obj.surrogate)
            if keras_owner is None:
                # Não deveria acontecer se o arquivo existe, mas tratamos graciosamente
                print("Aviso: arquivo .h5 encontrado, mas não há onde anexar o modelo no surrogate.")
            else:
                setattr(
                    keras_owner,
                    keras_attr,
                    keras.models.load_model(model_path, compile=False)
                )

        print(f"FlowReconstruction carregado de {pkl_path}")
        if model_path.exists():
            print(f"Modelo Keras carregado de {model_path}")
        return obj
    
    def setattr(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

