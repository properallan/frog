import os
import json
import tensorflow as tf
#from ray import tune
from ray import train
import time
import dill

def setup_logical_device(mem_per_gpu_mb=2048):
    import tensorflow as tf
    """Configura 1 logical GPU com a quantidade de memória definida por trial"""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Nenhuma GPU detectada.")
        return None

    # Cria 1 logical GPU por trial
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=mem_per_gpu_mb)]
    )
    
    logical_gpus = tf.config.list_logical_devices('GPU')
    return logical_gpus[0]

    
class LRTensorBoardLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.lrs = []

    def on_epoch_end(self, epoch, logs=None):
        # Get current learning rate (handles schedules too)
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except:
            step = self.model.optimizer.iterations
            lr = self.model.optimizer.learning_rate 
            lr = float(tf.keras.backend.get_value(lr(step)))
        self.lrs.append(lr)

        # Log to TensorBoard
        with self.writer.as_default():
            tf.summary.scalar('lr', data=lr, step=epoch)
            self.writer.flush()




def trainable(config, other_params={}):
    from frog.flow_reconstruction import FlowReconstruction
    from frog.metrics import NRMSE, R2, MAPE, MAXPE, MAE, MSE
    from frog.datahandler import DataHandlerNpz
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
    from frog.transformers import IdentityTransformer, MeanCentering, SliceMeanCentering
    from frog.normalization import PhysicalNormalizer, SliceMinMaxScaler, SliceMaxAbsScaler

    from sklearn.decomposition import TruncatedSVD, IncrementalPCA, PCA, KernelPCA, SparsePCA, MiniBatchSparsePCA
    from frog.neuralnetwork import NeuralNetwork
    import numpy as np
    import tensorflow as tf
    import ray
    from collections.abc import Iterable
    from frog.neuralnetwork import TuneReporterCallback

    def set_callbacks(other_params, tensorboard_logs_dir, model_dir, search_space):
        
        callbacks = []

        # Add callback para earling stopping
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            **other_params['early_stopping']    
        )
        callbacks.append(earlystop_callback)

        
        # Add callback para learning rate schedulers
        if 'learning_rate_scheduler' in other_params.keys():
            try:
                "eval twice if it uses value from search_space[KEY]"
                other_params['learning_rate_scheduler'] = eval(eval(other_params['learning_rate_scheduler']))
            except:
                "learning rate scheduler already in correct format"
                pass

            schedulers = []
            for learning_rate_scheduler_name in other_params['learning_rate_scheduler']:
                from frog.neuralnetwork import ReduceLROnPlateau, IncreaseLROnImprovement, EpochRangeReduceLROnPlateau, WarmupCosineDecay

                for key, val in other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name].items():
                    if type(val) == str:
                        try:
                            value = eval(val)
                        except:
                            value = val
                        other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name][key] = value

                schedulers.append(
                    eval(learning_rate_scheduler_name)(
                            **other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name]
                    )
                )
            callbacks.extend(schedulers)

            callbacks.append(LRTensorBoardLogger(tensorboard_logs_dir))

        return callbacks

    # Configura 1 logical GPU por trial
    #logical_gpu = setup_logical_device(mem_per_gpu_mb=other_params['mem_per_gpu_mb'])
    #tf.config.set_visible_devices([logical_gpu], 'GPU')

    trial_id = ray.train.get_context().get_trial_id()
    trial_dir = ray.train.get_context().get_trial_dir()
    
    search_space = config
    
    # load data
    training_X = DataHandlerNpz(other_params['TRAINING_X'])
    test_X = DataHandlerNpz(other_params['TEST_X'])
    validation_X = DataHandlerNpz(other_params['VALIDATION_X'])

    training_y = DataHandlerNpz(other_params['TRAINING_y'])
    test_y = DataHandlerNpz(other_params['TEST_y'])
    validation_y = DataHandlerNpz(other_params['VALIDATION_y'])
    
    # filter variables
    training_X = training_X[other_params['LF_VARIABLES']]
    test_X = test_X[other_params['LF_VARIABLES']]
    validation_X = validation_X[other_params['LF_VARIABLES']]

    training_y = training_y[other_params['HF_VARIABLES']]
    test_y = test_y[other_params['HF_VARIABLES']]
    validation_y = validation_y[other_params['HF_VARIABLES']]

    VALIDATION_DATA = (validation_X, validation_y)

    X_rom = eval(other_params['X_rom'])
    y_rom = eval(other_params['y_rom'])
    
    
    initial_epoch = 0
    metrics_dict = other_params['metrics']

    if 'kfold_cross_validation' in other_params.keys():
        from sklearn.model_selection import KFold

        kf = KFold(
            **other_params['kfold_cross_validation']
            )

        
        for k, (train_index, test_index) in enumerate(kf.split(training_X)):
            fold_metrics = {}
            X_train, X_test = training_X[train_index], training_X[test_index]
            y_train, y_test = training_y[train_index], training_y[test_index]

            
            model_dir = os.path.join(trial_dir, f'fold_{k}', "tensorflow_model")
            tensorboard_logs_dir = os.path.join(trial_dir, f'fold_{k}', "tensorboard_logs")
            fr_model_dir = os.path.join(trial_dir,  f'fold_{k}', "fr_model")

            if other_params['regressor'].split('(')[0] == 'NeuralNetwork':
                def replace_last(text, old, new):
                    parts = text.rsplit(old, 1)
                    return new.join(parts)
                fit_kwargs_subset = {k: other_params['fit_kwargs'][k] for k in ['regressor__epochs', 'regressor__batch_size', 'regressor__dataset_size'] if k in other_params['fit_kwargs']}
                fit_kwargs_subset.update({'regressor__dataset_size': X_train.shape[0]})
                other_params_regressor = replace_last(other_params['regressor'], ')', f", fit_kwargs={fit_kwargs_subset})")


            regressor = eval(other_params_regressor)
            regressor.rom = y_rom

            # Restaurar modelo se já tiver salvo
            
            if os.path.exists(model_dir+"/model.keras"):
                
                regressor.model = tf.keras.models.load_model(model_dir+"/model.keras", compile=True)
                regressor.compile()
            
                with open(os.path.join(tensorboard_logs_dir, "current_epoch.txt"), "r") as f:
                    initial_epoch = int(f.readline().strip())    
               
            else:
                print("Não foi possível restaurar o modelo")
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(fr_model_dir, exist_ok=True)
                os.makedirs(tensorboard_logs_dir, exist_ok=True)
            

            surrogate = Pipeline([
                ('regressor', regressor),
            ])

            builder = eval(other_params['model_builder'])            
            fr = builder(X_rom=X_rom, y_rom=y_rom, surrogate=surrogate)

            callbacks = set_callbacks(
                other_params=other_params,
                tensorboard_logs_dir=tensorboard_logs_dir, 
                model_dir=model_dir, 
                search_space=search_space)
            
            callbacks.append(TuneReporterCallback(
                    log_dir=tensorboard_logs_dir,
                    fr_model=fr,
                    test_X=X_test,
                    test_y=y_test,
                    metrics_dict=metrics_dict, 
                    model_dir=model_dir,
                    kfold_iteration=k
                ))

            fit_kwargs = dict( 
                regressor__callbacks=callbacks,
                regressor__verbose=0
            )
            
            for key, val in other_params['fit_kwargs'].items():
                if type(val) == str:
                    other_params['fit_kwargs'][key] = eval(val)

            fit_kwargs.update(other_params['fit_kwargs'])
            fit_kwargs.update({'regressor__initial_epoch': initial_epoch})

            # if not regressor.model._is_compiled:
            #     # Corrige problema de modelo nao compilao ao recuperar otimizacao de hiperparametros
            #     regressor.compilte()
            #     surrogate = Pipeline([
            #         ('regressor', regressor),
            #     ])
            #     fr.surrogate = surrogate

           
            fr.fit(X=X_train, y=y_train, **fit_kwargs)
            prediction = fr.predict(X_test)
            ground_truth = y_test
            


            fold_metrics['kfold_iteration'] = k
                  
            for key, value in metrics_dict.items():
                if key not in fold_metrics.keys():
                    fold_metrics[key] = []
               
                fold_metrics[key].append(float(eval(value)(ground_truth, prediction)))


            with open(os.path.join(tensorboard_logs_dir, 'history.json'), "r") as f:
                 history = json.load(f)
   
            #fold_metrics.update({k:v[-1] if (isinstance(v, Iterable) and v!=[]) else 0 for k,v in history.items()})
            
            # with open(os.path.join(fr_model_dir, 'fr_model.pkl'), "wb") as f:
            #     dill.dump(fr, f)  # Salva o modelo como pickle

            from tensorflow.keras.models import save_model
        
            if model_dir is not None:
                    save_model(
                    model=fr.surrogate.named_steps['regressor'].model,
                    filepath=model_dir+"/model.keras",
                    include_optimizer=True,   # evita problemas com LR schedules customizados
                    #save_format="tf"          # força SavedModel (pasta)
                )

        metrics = {key: np.mean(values) if key!='kfold_iteration' else values for key, values in fold_metrics.items()}

        metrics.update({'kfold_iteration': fold_metrics['kfold_iteration']})
        with open(os.path.join(trial_dir, 'metrics.json'), "w") as f:
            json.dump(metrics, f)
    else:
        
        model_dir = os.path.join(trial_dir, "tensorflow_model")
        tensorboard_logs_dir = os.path.join(trial_dir, "tensorboard_logs")
        fr_model_dir = os.path.join(trial_dir, "fr_model")

        if other_params['regressor'].split('(')[0] == 'NeuralNetwork':
            def replace_last(text, old, new):
                parts = text.rsplit(old, 1)
                return new.join(parts)
            fit_kwargs_subset = {k: other_params['fit_kwargs'][k] for k in ['regressor__epochs', 'regressor__batch_size', 'regressor__dataset_size'] if k in other_params['fit_kwargs']}
            fit_kwargs_subset.update({'regressor__dataset_size': training_X.shape[0]})
            other_params_regressor = replace_last(other_params['regressor'], ')', f", fit_kwargs={fit_kwargs_subset})")

        regressor = eval(other_params_regressor)
        regressor.rom = y_rom

        # Restaurar modelo se já tiver salvo
        if os.path.exists(model_dir+"/model.keras"):
            regressor.model = tf.keras.models.load_model(model_dir+"/model.keras", compile=True)
            regressor.compile()
                
            with open(os.path.join(tensorboard_logs_dir, "current_epoch.txt"), "r") as f:
                initial_epoch = int(f.readline().strip())    
                
        else:
            print("Não foi possível restaurar o modelo")
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(fr_model_dir, exist_ok=True)
            os.makedirs(tensorboard_logs_dir, exist_ok=True)
        
        surrogate = Pipeline([
            ('regressor', regressor),
        ])

        builder = eval(other_params['model_builder'])
        fr = builder(X_rom=X_rom, y_rom=y_rom, surrogate=surrogate)

        callbacks = set_callbacks(
                other_params=other_params,
                tensorboard_logs_dir=tensorboard_logs_dir, 
                model_dir=model_dir, 
                search_space=search_space,
                )
        
        callbacks.append(TuneReporterCallback(
                    log_dir=tensorboard_logs_dir,
                    fr_model=fr,
                    test_X=test_X,
                    test_y=test_y,
                    metrics_dict=metrics_dict, 
                    model_dir=model_dir
                ))

        
        fit_kwargs = dict( 
            regressor__callbacks=callbacks,
            regressor__verbose=0
        )
        
        for key, val in other_params['fit_kwargs'].items():
            if type(val) == str:
                other_params['fit_kwargs'][key] = eval(val)

        fit_kwargs.update(other_params['fit_kwargs'])
        fit_kwargs.update({'regressor__initial_epoch': initial_epoch})

        # if not regressor.model._is_compiled:
        #     # Corrige problema de modelo nao compilao ao recuperar otimizacao de hiperparametros
        #     regressor.compilte()
        #     surrogate = Pipeline([
        #         ('regressor', regressor),
        #     ])
        #     fr.surrogate = surrogate

        #print('EVALUATE WITHOUT KFOLD')
        fr.fit(X=training_X, y=training_y, **fit_kwargs) 
        prediction = fr.predict(test_X)
        ground_truth = test_y

        metrics = {}
        for key, value in metrics_dict.items():
            metrics[key] = float(eval(value)(ground_truth, prediction))

        with open(os.path.join(tensorboard_logs_dir, 'history.json'), "r") as f:
             history = json.load(f)
        
        metrics.update({k:v[-1] if isinstance(v, Iterable) else 0 for k,v in history.items()})
       
        # with open(os.path.join(fr_model_dir, 'fr_model.pkl'), "wb") as f:
        #     dill.dump(fr, f)  # Salva o modelo como pickle

        with open(os.path.join(trial_dir, 'metrics.json'), "w") as f:
            json.dump(metrics, f)
       

        from tensorflow.keras.models import save_model

        if model_dir is not None:
                save_model(
                model=fr.surrogate.named_steps['regressor'].model,
                filepath=model_dir+"/model.keras",
                include_optimizer=True,   # evita problemas com LR schedules customizados
                #save_format="tf"          # força SavedModel (pasta)
            )

    
    return metrics