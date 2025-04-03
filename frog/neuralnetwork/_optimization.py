import os
import json
import tensorflow as tf
#from ray import tune
from ray import train

class TuneReporterCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir="logs"):
        super().__init__()
        self.log_dir = log_dir
        self.history = {"loss": [], "val_loss": []}
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Salva os logs no histórico
        self.history["loss"].append(logs.get("loss"))
        self.history["val_loss"].append(logs.get("val_loss"))

        # Reporta métricas para o Ray Tune
        #train.report(dict(loss=logs.get("loss"), val_loss=logs.get("val_loss")))

        # Registra métricas no TensorBoard
        with self.writer.as_default():
            for key, value in logs.items():
                tf.summary.scalar(key, value, step=epoch)
            self.writer.flush()

    def on_train_end(self, logs=None):
        # Salvar histórico de treinamento em um arquivo JSON
        history_path = os.path.join(self.log_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f)

class IncreaseLROnImprovement(tf.keras.callbacks.Callback):
    def __init__(self, factor=2, threshold=0.1, patience=3, monitor="val_loss"):
        super(IncreaseLROnImprovement, self).__init__()
        self.factor = factor  # Multiplicador do LR
        self.threshold = threshold  # Percentual mínimo de queda para considerar melhoria
        self.patience = patience  # Número de épocas consecutivas para aumentar LR
        self.monitor = monitor
        self.prev_loss = None
        self.wait = 0  # Contador de épocas consecutivas de melhora

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get(self.monitor)
        
        if current_loss is None:
            return
        
        if self.prev_loss is not None:
            loss_reduction = (self.prev_loss - current_loss) / self.prev_loss
            
            if loss_reduction > self.threshold:
                self.wait += 1  # Contar épocas consecutivas de melhora
                if self.wait >= self.patience:
                    old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                    new_lr = old_lr * self.factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    print(f"\nAumentando learning rate de {old_lr:.6f} para {new_lr:.6f} (epoch {epoch+1})")
                    self.wait = 0  # Resetar contador após ajuste
            else:
                self.wait = 0  # Resetar contador se não houve melhora suficiente

        self.prev_loss = current_loss

def trainable(config, other_params={}):
    from frog.flow_reconstruction import FlowReconstruction
    from frog.metrics import NRMSE, R2, MAPE, MAXPE, MAE, MSE
    from frog.datahandler import DataHandlerNpz
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
    from frog.transformers import IdentityTransformer, MeanCenteringTransformer
    
    from sklearn.decomposition import TruncatedSVD, IncrementalPCA, PCA, KernelPCA, SparsePCA, MiniBatchSparsePCA
    from frog.neuralnetwork import NeuralNetwork
    import numpy as np
    import tensorflow as tf
    import ray
    
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

    # X_scaler = eval(other_params['X_scaler'])
    # y_scaler = eval(other_params['y_scaler'])

    #X_latent_scaler = eval(other_params['X_latent_scaler'])
    #y_latent_scaler = eval(other_params['y_latent_scaler'])

    # X_reducer = eval(other_params['X_reducer'])
    # y_reducer = eval(other_params['y_reducer'])

    # X_rom = Pipeline([X_scaler, X_reducer])#, X_latent_scaler])
    # y_rom = Pipeline([y_scaler, y_reducer])#, y_latent_scaler])

    X_rom = eval(other_params['X_rom'])
    y_rom = eval(other_params['y_rom'])


    #fr = FRNNBuilder(**{**config, **other_params})
    #fr = builder(**{**config, **other_params})

    callbacks = []


    # Add callback para earling stopping
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        **other_params['early_stopping']    
    )
    callbacks.append(earlystop_callback)

    if 'learning_rate_scheduler' in other_params.keys():
        schedulers = []
        for learning_rate_scheduler_name in other_params['learning_rate_scheduler']:
            if learning_rate_scheduler_name.upper() == 'ReduceLROnPlateau'.upper():
                schedulers.append(
                    tf.keras.callbacks.ReduceLROnPlateau(
                        **other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name]
                    )
                )

            elif learning_rate_scheduler_name.upper() == 'IncreaseLROnImprovement'.upper():
                schedulers.append(
                    IncreaseLROnImprovement(
                        **other_params['learning_rate_scheduler_kwargs'][learning_rate_scheduler_name]
                    )
                )

        callbacks.extend(schedulers)

    # Add callback para learning rate scheduler
    # if 'learning_rate_scheduler' in other_params.keys():
    #     if other_params['learning_rate_scheduler'].upper() == 'ReduceLROnPlateau'.upper() :
    #         lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    #                 #monitor=other_params['learning_rate_scheduler']['monitor'],         # Monitorar a métrica de validação
    #                 #factor=other_params['learning_rate_scheduler']['factor'],                 # Reduzir a taxa de aprendizado por um fator de 10
    #                 #patience=other_params['learning_rate_scheduler']['patience'],                 # Quantas épocas sem melhoria antes de reduzir o lr
    #                 #min_lr=other_params['learning_rate_scheduler']['min_lr']                 # Taxa mínima para a qual o lr pode ser reduzido
    #             **other_params['learning_rate_scheduler_kwargs']
    #             )
    #         callbacks.append(lr_scheduler)       

    # Add callback para reportar losses
    experiment_dir = os.path.join(ray.train.get_context().get_trial_dir(), "tensorboard_logs")
    os.makedirs(experiment_dir, exist_ok=True)
    callbacks.append(TuneReporterCallback(log_dir=experiment_dir))
    
    regressor = eval(other_params['regressor'])

    # Criar um callback para printar o learning rate
    # print_lr_callback = tf.keras.callbacks.LambdaCallback(
    #     on_epoch_end=lambda epoch, logs:
    #       #print(f"\nEpoch {epoch+1} | LR: {tf.keras.backend.get_value(regressor.optimizer.lr):.6f} | Loss: {logs['loss']:.6f} | Val Loss: {logs.get('val_loss', 'N/A'):.6f}")
    #       print(f" - lr: {tf.keras.backend.get_value(regressor.optimizer.lr):.4f}")
    # )

    #callbacks.append(print_lr_callback)

    fit_kwargs = dict( 
        regressor__callbacks=callbacks,
    )

    surrogate = Pipeline([
        ('regressor', regressor),
    ])

    
    for key, val in other_params['fit_kwargs'].items():
        if type(val) == str:
            other_params['fit_kwargs'][key] = eval(val)

    fit_kwargs.update(other_params['fit_kwargs'])

    builder = eval(other_params['model_builder'])
    fr = builder(X_rom=X_rom, y_rom=y_rom, surrogate=surrogate)
    
    

    #metrics_dict = eval_dict(other_params['metrics'])
    metrics_dict = other_params['metrics']

    if 'kfold_cross_validation' in other_params.keys():
        from sklearn.model_selection import KFold

        training_X = np.vstack((training_X, validation_X))
        training_y = np.vstack((training_y, validation_y))

        kf = KFold(
            #n_splits=K, shuffle=True, random_state=42
            **other_params['kfold_cross_validation']
            )


        fold_metrics = {}
        for train_index, val_index in kf.split(training_X):
            
            X_train, X_val = training_X[train_index], training_X[val_index]
            y_train, y_val = training_y[train_index], training_y[val_index]

            VALIDATION_DATA = (X_val, y_val)
            fit_kwargs.update({'regressor__validation_data': VALIDATION_DATA })

            fr = builder(X_rom=X_rom, y_rom=y_rom, surrogate=surrogate)
            fr.fit(X=X_train, y=y_train, **fit_kwargs)

            prediction = fr.predict(X_val)
            ground_truth = y_val


            for key, value in metrics_dict.items():
                if key not in fold_metrics.keys():
                    fold_metrics[key] = []
                fold_metrics[key].append(float(eval(value)(ground_truth, prediction)))

        metrics = {key: np.mean(values) for key, values in fold_metrics.items()}

    else:
        fr.fit(X=training_X, y=training_y, **fit_kwargs) 
        prediction = fr.predict(validation_X)
        #ground_truth = test_y
        ground_truth = validation_y

        metrics = {}
        for key, value in metrics_dict.items():
            metrics[key] = float(eval(value)(ground_truth, prediction))
    
    import dill
    #from ray.air import session
    # Criando diretório para salvar o checkpoint
    trial_dir = ray.train.get_context().get_trial_dir()

    with open(os.path.join(trial_dir, 'model.pkl'), "wb") as f:
        dill.dump(fr, f)  # Salva o modelo como pickle

    return metrics