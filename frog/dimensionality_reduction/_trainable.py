from frog.flow_reconstruction import FlowReconstruction

def trainable(config, other_params={}, model_builder=FlowReconstruction):
    import yaml
    from frog.flow_reconstruction import FlowReconstruction

    from pathlib import Path
    from frog.metrics import NRMSE, R2, MAPE, MAXPE, MAE, MSE
    import numpy as np
    import pandas as pd
    from ray import train, tune
    import os   
    from frog.doe import dict_to_array_and_index, array_and_index_to_dict
    from sklearn.preprocessing import MinMaxScaler

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
    from sklearn.decomposition import TruncatedSVD, IncrementalPCA, PCA, KernelPCA
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, DotProduct, WhiteKernel, ConstantKernel, RationalQuadratic
    from frog.transformers import IdentityTransformer
    from sklearn.linear_model import LinearRegression
    from frog.normalization import RowScaler, RowScalerNpz
    from frog.datahandler import get_snapshot_end_index, DataHandlerNpz

    # load data
    training_X = DataHandlerNpz(other_params['TRAINING_X'])
    training_y = DataHandlerNpz(other_params['TRAINING_y'])

    test_X = DataHandlerNpz(other_params['TEST_X'])
    test_y = DataHandlerNpz(other_params['TEST_y'])

    # filter variables
    training_X = training_X[other_params['LF_VARIABLES']]
    test_X = test_X[other_params['LF_VARIABLES']]

    training_y = training_y[other_params['HF_VARIABLES']]
    test_y = test_y[other_params['HF_VARIABLES']]

    X_scaler = eval(other_params['X_scaler'])
    y_scaler = eval(other_params['y_scaler'])

    X_reducer = eval(other_params['X_reducer'])
    y_reducer = eval(other_params['y_reducer'])

    X_rom = Pipeline([X_scaler, X_reducer])
    y_rom = Pipeline([y_scaler, y_reducer])

    kernel = eval(other_params['kernel'])   

    regressor = eval(other_params['regressor'])

    surrogate = Pipeline([
        ('regressor', regressor),
    ])

    fr = model_builder(X_rom=X_rom, y_rom=y_rom, surrogate=surrogate, kernel=kernel, surrogate_kwargs={})

    fr.fit(X=training_X, y=training_y)

    prediction = fr.predict(test_X)
    ground_truth = test_y

    metrics_dict = other_params['metrics']

    metrics = {}
    for key, value in metrics_dict.items():
        metrics[key] = eval(value)(ground_truth, prediction)

    return metrics