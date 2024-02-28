def trainable(config, other_params={}):
    from frog.flow_reconstruction import FlowReconstruction
    from frog.metrics import NRMSE, R2, MAPE, MAXPE, MAE, MSE
    from frog.datahandler import DataHandlerNpz
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
    
    from sklearn.decomposition import TruncatedSVD, IncrementalPCA, PCA, KernelPCA, SparsePCA, MiniBatchSparsePCA
    from frog.neuralnetwork import NeuralNetwork
    import numpy as np
    import tensorflow as tf
    
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

    X_scaler = eval(other_params['X_scaler'])
    y_scaler = eval(other_params['y_scaler'])

    X_reducer = eval(other_params['X_reducer'])
    y_reducer = eval(other_params['y_reducer'])

    X_rom = Pipeline([X_scaler, X_reducer])
    y_rom = Pipeline([y_scaler, y_reducer])

    #fr = FRNNBuilder(**{**config, **other_params})
    #fr = builder(**{**config, **other_params})

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        **other_params['early_stopping']    
    )

    fit_kwargs = dict( 
        regressor__callbacks=[earlystop_callback],
    )
    
    regressor = eval(other_params['regressor'])

    surrogate = Pipeline([
        ('regressor', regressor),
    ])

    
    for key, val in other_params['fit_kwargs'].items():
        if type(val) == str:
            other_params['fit_kwargs'][key] = eval(val)

    fit_kwargs.update(other_params['fit_kwargs'])

    builder = eval(other_params['model_builder'])
    fr = builder(X_rom=X_rom, y_rom=y_rom, surrogate=surrogate)
    
    fr.fit(X=training_X, y=training_y, **fit_kwargs) 
    prediction = fr.predict(test_X)
    ground_truth = test_y

    #metrics_dict = eval_dict(other_params['metrics'])
    metrics_dict = other_params['metrics']

    metrics = {}
    for key, value in metrics_dict.items():
        metrics[key] = float(eval(value)(ground_truth, prediction))
   
    return metrics