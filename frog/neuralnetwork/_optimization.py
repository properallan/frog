import os
import json
import tensorflow as tf
#from ray import tune
from ray import train
import time
import dill


    
class LRTensorBoardLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.lrs = []

    def on_epoch_end(self, epoch, logs=None):
        # Get current learning rate (handles schedules too)
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        except:
            step = self.model.optimizer.iterations
            lr = self.model.optimizer.lr    
            lr = float(tf.keras.backend.get_value(lr(step)))
        self.lrs.append(lr)

        # Log to TensorBoard
        with self.writer.as_default():
            tf.summary.scalar('lr', data=lr, step=epoch)
            self.writer.flush()

from frog.neuralnetwork import TuneReporterCallback
# class TuneReporterCallback(tf.keras.callbacks.Callback):
#     def __init__(self, log_dir="logs"):
#         super().__init__()
#         self.log_dir = log_dir
#         self.history = {"loss": [], "val_loss": [], "lr": []}
#         self.writer = tf.summary.create_file_writer(log_dir)

#     def on_train_begin(self, logs=None):
        
#         self.start_time = time.time()

#     def on_epoch_end(self, epoch, logs=None):
#         if logs is None:
#             logs = {}

#         try:
#             lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
#         except:
#             stehup = self.model.optimizer.iterations
#             lr = self.model.optimizer.lr    
#             lr = float(tf.keras.backend.get_value(lr(step)))
        
#         logs['lr'] = lr

#         # Salva os logs no histórico
#         self.history["loss"].append(logs.get("loss"))
#         self.history["val_loss"].append(logs.get("val_loss"))
#         self.history["lr"].append(logs.get("lr"))

#         # Reporta métricas para o Ray Tune
#         #train.report(dict(loss=logs.get("loss"), val_loss=logs.get("val_loss")))

#         # Registra métricas no TensorBoard
#         with self.writer.as_default():
        
#             for key, value in logs.items():
#                 tf.summary.scalar(key, value, step=epoch)
#             self.writer.flush()

#         if epoch % 10 == 0:
#             elapsed = time.time() - self.start_time
#             it_s = (epoch + 1) / elapsed
#             try:
#                 lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
#             except:
#                 step = self.model.optimizer.iterations
#                 lr = self.model.optimizer.lr    
#                 lr = float(tf.keras.backend.get_value(lr(step)))
#             #print(f"[Epoch {epoch+1}] Iterações por segundo: {it_s:.2f}")
#             print(f"Epoch {epoch}: loss={logs['loss']:>4.4f}, val_loss={logs['val_loss']:4>.4f}, lr={lr:>4.4f}, {it_s:>4.4f} it/s")

#     def on_train_end(self, logs=None):
#         # Salvar histórico de treinamento em um arquivo JSON
#         history_path = os.path.join(self.log_dir, "history.json")
#         with open(history_path, "w") as f:
#             json.dump(self.history, f)



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

    callbacks = []

    # Add callback para earling stopping
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        **other_params['early_stopping']    
    )
    callbacks.append(earlystop_callback)

    
    # Add callback para learning rate schedulers
    if 'learning_rate_scheduler' in other_params.keys():
        #print(other_params['learning_rate_scheduler'])
        try:
            "eval twice if it uses value from search_space[KEY]"
            other_params['learning_rate_scheduler'] = eval(eval(other_params['learning_rate_scheduler']))
        except:
            "learning rate scheduler already in correct format"
            pass

        #print(other_params['learning_rate_scheduler'])
        

        schedulers = []
        for learning_rate_scheduler_name in other_params['learning_rate_scheduler']:
            from frog.neuralnetwork import ReduceLROnPlateau, IncreaseLROnImprovement, EpochRangeReduceLROnPlateau, WarmupCosineDecay

            for key, val in other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name].items():
                if type(val) == str:
                    other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name][key] = eval(val)
                #from frog.utils import eval_dict
                #other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name] = eval_dict(other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name])
            schedulers.append(
                eval(learning_rate_scheduler_name)(
                         **other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name]
                )
            )
            # if learning_rate_scheduler_name.upper() == 'ReduceLROnPlateau'.upper():
            #      schedulers.append(
            #          tf.keras.callbacks.ReduceLROnPlateau(
            #              **other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name]
            #          )
            #      )
            # elif learning_rate_scheduler_name.upper() == 'WarmupCosineDecay'.upper():
            #      schedulers.append(
            #          WarmupCosineDecay(
            #              **other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name]
            #          )
            #      )
            # elif learning_rate_scheduler_name.upper() == 'IncreaseLROnImprovement'.upper():
            #      schedulers.append(
            #          IncreaseLROnImprovement(
            #              **other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name]
            #          )
            #      )

            # elif learning_rate_scheduler_name.upper() == 'IncreaseLROnImprovement'.upper():
            #     schedulers.append(
            #         IncreaseLROnImprovement(
            #             **other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name]
            #         )
            #     )
            # else: #learning_rate_scheduler_name.upper() == 'IncreaseLROnImprovement'.upper():
            #     schedulers.append(
            #         eval(learning_rate_scheduler_name)(
            #             **other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name]
            #         )
            #     )


            #schedulers.append(eval(learning_rate_scheduler))

        #print(schedulers)
        callbacks.extend(schedulers)
    
    
    initial_epoch = 0
    metrics_dict = other_params['metrics']

    if 'kfold_cross_validation' in other_params.keys():
        from sklearn.model_selection import KFold

        kf = KFold(
            **other_params['kfold_cross_validation']
            )

        fold_metrics = {}
        for k, (train_index, test_index) in enumerate(kf.split(training_X)):
            X_train, X_test = training_X[train_index], training_X[test_index]
            y_train, y_test = training_y[train_index], training_y[test_index]

            fold_dir = os.path.join(trial_dir, f'fold_{k}')
            model_dir = os.path.join(trial_dir, f'fold_{k}', "tensorflow_model")
            tensorboard_logs_dir = os.path.join(trial_dir, f'fold_{k}', "tensorboard_logs")
            fr_model_dir = os.path.join(trial_dir,  f'fold_{k}', "fr_model")

            if other_params['regressor'].split('(')[0] == 'NeuralNetwork':
                def replace_last(text, old, new):
                    parts = text.rsplit(old, 1)
                    return new.join(parts)
                fit_kwargs_subset = {k: other_params['fit_kwargs'][k] for k in ['regressor__epochs', 'regressor__batch_size', 'regressor__dataset_size'] if k in other_params['fit_kwargs']}
                fit_kwargs_subset.update({'regressor__dataset_size': X_train.shape[0]})
                #print(f"fit_kwargs={fit_kwargs_subset}")
                other_params_regressor = replace_last(other_params['regressor'], ')', f", fit_kwargs={fit_kwargs_subset})")
                #print(other_params_regressor)

            regressor = eval(other_params_regressor)
            regressor.rom = y_rom

            # Restaurar modelo se já tiver salvo
            
            if os.path.exists(model_dir):
                try:
                    with tf.device('/GPU:0'):
                        regressor.model = tf.keras.models.load_model(model_dir, compile=False)
                        regressor.compile()
                    
                    with open(os.path.join(tensorboard_logs_dir, "current_epoch.txt"), "r") as f:
                        initial_epoch = int(f.readline().strip())    
                except:
                    print("Não foi possível restaurar o modelo")
            else:
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(fr_model_dir, exist_ok=True)
                os.makedirs(tensorboard_logs_dir, exist_ok=True)
            

            surrogate = Pipeline([
                ('regressor', regressor),
            ])

            builder = eval(other_params['model_builder'])            
            fr = builder(X_rom=X_rom, y_rom=y_rom, surrogate=surrogate)

            callbacks.append(TuneReporterCallback(
                log_dir=tensorboard_logs_dir,
                # fr_model=fr,
                # test_X=X_test,
                # test_y=y_test,
                # metrics_dict=metrics_dict, 
                model_dir=model_dir
            ))
            callbacks.append(LRTensorBoardLogger(tensorboard_logs_dir))

            fit_kwargs = dict( 
                regressor__callbacks=callbacks,
                regressor__verbose=0
            )
            
            for key, val in other_params['fit_kwargs'].items():
                if type(val) == str:
                    other_params['fit_kwargs'][key] = eval(val)

            fit_kwargs.update(other_params['fit_kwargs'])
            fit_kwargs.update({'regressor__initial_epoch': initial_epoch})

            fr.fit(X=X_train, y=y_train, **fit_kwargs)
            prediction = fr.predict(X_test)

            ground_truth = y_test

            instance_metrics = {}
            for key, value in metrics_dict.items():
                if key not in fold_metrics.keys():
                    fold_metrics[key] = []
                instance_metrics[key] = float(eval(value)(ground_truth, prediction))
                fold_metrics[key].append(instance_metrics[key])

            with open(os.path.join(tensorboard_logs_dir, 'history.json'), "r") as f:
                 history = json.load(f)
            fold_metrics.update({k:v[-1] if isinstance(v, Iterable) else 0 for k,v in history.items()})

            with open(os.path.join(fr_model_dir, 'fr_model.pkl'), "wb") as f:
                dill.dump(fr, f)  # Salva o modelo como pickle

            # save_model(
            #     model=regressor.model,
            #     filepath=model_dir,
            #     include_optimizer=True,  # evita problemas com LR schedules customizados
            #     save_format="tf"          # força SavedModel (pasta)
            # )

            # with open(os.path.join(fold_dir, 'metrics.json'), "w") as f:
            #     json.dump(instance_metrics, f)

        metrics = {key: np.mean(values) for key, values in fold_metrics.items()}

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
            #print(f"fit_kwargs={fit_kwargs_subset}")
            other_params_regressor = replace_last(other_params['regressor'], ')', f", fit_kwargs={fit_kwargs_subset})")
            #print(other_params_regressor)

        regressor = eval(other_params_regressor)
        regressor.rom = y_rom

        # Restaurar modelo se já tiver salvo
        if os.path.exists(model_dir):
            with tf.device('/GPU:0'):
                regressor.model = tf.keras.models.load_model(model_dir, compile=False)    
                regressor.compile()

            with open(os.path.join(tensorboard_logs_dir, "current_epoch.txt"), "r") as f:
                initial_epoch = int(f.readline().strip())   
            # try:
            #     #print(f"[{trial_id}] Restaurando modelo de {model_dir}")
            #     regressor.model = tf.keras.models.load_model(model_dir)
            #     #regressor.compile()
                
            #     with open(os.path.join(tensorboard_logs_dir, "current_epoch.txt"), "w") as f:
            #         initial_epoch = int(f.readline().strip())    
            # except:
            #     initial_epoch = 0
            #     print("Não foi possível restaurar o modelo")
        else:
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(fr_model_dir, exist_ok=True)
            os.makedirs(tensorboard_logs_dir, exist_ok=True)
        
        
    
        surrogate = Pipeline([
            ('regressor', regressor),
        ])

        builder = eval(other_params['model_builder'])
        fr = builder(X_rom=X_rom, y_rom=y_rom, surrogate=surrogate)

        callbacks.append(TuneReporterCallback(
                log_dir=tensorboard_logs_dir,
                # fr_model=fr,
                # test_X=test_X,
                # test_y=test_y,
                # metrics_dict=metrics_dict, 
                model_dir=model_dir
        ))
        callbacks.append(LRTensorBoardLogger(tensorboard_logs_dir))

        fit_kwargs = dict( 
            regressor__callbacks=callbacks,
            regressor__verbose=0
        )
        
        for key, val in other_params['fit_kwargs'].items():
            if type(val) == str:
                other_params['fit_kwargs'][key] = eval(val)

        fit_kwargs.update(other_params['fit_kwargs'])
        fit_kwargs.update({'regressor__initial_epoch': initial_epoch})

        fr.fit(X=training_X, y=training_y, **fit_kwargs) 
        prediction = fr.predict(test_X)
        ground_truth = test_y

        metrics = {}
        for key, value in metrics_dict.items():
            metrics[key] = float(eval(value)(ground_truth, prediction))

        with open(os.path.join(tensorboard_logs_dir, 'history.json'), "r") as f:
             history = json.load(f)
        
        metrics.update({k:v[-1] if isinstance(v, Iterable) else 0 for k,v in history.items()})
       
        with open(os.path.join(fr_model_dir, 'fr_model.pkl'), "wb") as f:
            dill.dump(fr, f)  # Salva o modelo como pickle

        # Salvar como SavedModel
        # Apenas o modelo "puro", sem histórico nem lixo extra

        # save_model(
        #     model=regressor.model,
        #     filepath=model_dir,
        #     include_optimizer=True,   # evita problemas com LR schedules customizados
        #     save_format="tf"          # força SavedModel (pasta)
        # )
        # print(f"model saved at {model_dir}")

        with open(os.path.join(trial_dir, 'metrics.json'), "w") as f:
            json.dump(metrics, f)

    return metrics