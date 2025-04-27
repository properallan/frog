import typer
from typing import Annotated
import time  # Importando o módulo time para medir o tempo de execução

app = typer.Typer()

@app.command()
def train(
    config_file: Annotated[str, typer.Argument(help='YAML configuration file to train the gaussian process.')],
):
    import yaml
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
    from sklearn.decomposition import TruncatedSVD, IncrementalPCA, PCA, KernelPCA, SparsePCA, MiniBatchSparsePCA

    from frog.flow_reconstruction import FlowReconstruction
    from frog.datahandler import DataHandlerNpz
    from frog.normalization import PhysicalNormalizer, SliceMinMaxScaler, SliceMaxAbsScaler
    from frog.metrics import NRMSE, R2, MAPE, MAXPE, MAE, MSE
    from frog.transformers import IdentityTransformer, MeanCentering, SliceMeanCentering
    from frog.utils import create_clean_directory

    import os
    import dill
    import os
    import pandas as pd
    import shutil
    # Marcar o tempo de início da execução
    start_time = time.time()

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    params = config['params']

    params['regressor'] = params['regressor']
    params['X_rom'] = eval(params['X_rom'])
    params['y_rom'] = eval(params['y_rom'])

    # Load data
    training_X = DataHandlerNpz(params['TRAINING_X'])
    training_y = DataHandlerNpz(params['TRAINING_y'])

    test_X = DataHandlerNpz(params['TEST_X'])
    test_y = DataHandlerNpz(params['TEST_y'])

    # Filter variables
    training_X = training_X[params['LF_VARIABLES']]
    test_X = test_X[params['LF_VARIABLES']]

    training_y = training_y[params['HF_VARIABLES']]
    test_y = test_y[params['HF_VARIABLES']]

    if 'kfold_cross_validation' in config.keys():
        from sklearn.model_selection import KFold
        import numpy as np

        create_clean_directory(config['save_results'])

        kfold = KFold(**config['kfold_cross_validation'])  # Exemplo com 5 splits

        fold_metrics = []  # Lista para armazenar as métricas de cada fold
        
        for k, (train_index, val_index) in enumerate(kfold.split(training_X)):
            start = time.perf_counter()
            save_path = os.path.join(config['save_results'],f'fold_{k}')
            create_clean_directory(save_path)

            metrics_csv = os.path.join(save_path,'metrics.csv')
            model_file = os.path.join(save_path,'fr_model.pkl')
            config_file_copy = os.path.join(save_path,'config.yaml')

            X_train, X_test = training_X[train_index], training_X[val_index]
            y_train, y_test = training_y[train_index], training_y[val_index]

            regressor = eval(params['regressor'])

            surrogate = Pipeline([('regressor', regressor)])

            # Criar e treinar o modelo
            model_builder = eval(config['fr_model_builder'])
            fr = model_builder(X_rom=params['X_rom'], y_rom=params['y_rom'], surrogate=surrogate, surrogate_kwargs={})

            fr.fit(X=X_train, y=y_train)

            # Fazer a previsão
            prediction = fr.predict(X_test)

            # Calcular as métricas (exemplo com MSE)
            metrics_dict = params['metrics']
            metrics = {}
            for key, value in metrics_dict.items():
                metrics[key] = eval(value)(y_test, prediction)
            fold_metrics.append(metrics)

            fold_metrics_df = pd.DataFrame(metrics, index=[0])
            fold_metrics_df.to_csv(metrics_csv)

            # Salvando o modelo com dill
            with open(model_file, 'wb') as f:
                dill.dump(fr, f)  # Salva o modelo completo, incluindo o regressor e o pipeline
                print(f"Modelo salvo em {model_file}")
            
            shutil.copy(config_file, config_file_copy)

            end = time.perf_counter()
            training_time = end - start

            with open(os.path.join(save_path, 'training_time.txt'), "w") as f:
                f.write(str(training_time))


        start = time.perf_counter()

        save_path = os.path.join(config['save_results'])

        metrics_csv = os.path.join(save_path,'metrics.csv')
        fold_metrics_csv = os.path.join(save_path,'fold_metrics.csv')
        model_file = os.path.join(save_path,'fr_model.pkl')
        config_file_copy = os.path.join(save_path,'config.yaml')

        # Calcular a média das métricas de todos os folds
        avg_metrics = {key: np.mean([fold[key] for fold in fold_metrics]) for key in fold_metrics[0]}

        # Treinar o modelo final com o conjunto completo de dados (sem K-fold)
        fr_final = model_builder(X_rom=params['X_rom'], y_rom=params['y_rom'], surrogate=surrogate, surrogate_kwargs={})
        fr_final.fit(X=training_X, y=training_y)

        # Realizar a previsão no conjunto de teste
        final_prediction = fr_final.predict(test_X)
        final_metrics = {}
        for key, value in metrics_dict.items():
            final_metrics[key] = eval(value)(test_y, final_prediction)

        fold_metrics_df = pd.DataFrame(fold_metrics)
        fold_metrics_df.to_csv(fold_metrics_csv)

        final_metrics_df = pd.DataFrame(final_metrics, index=[0])
        final_metrics_df.to_csv(metrics_csv)

        # Salvando o modelo com dill
        with open(model_file, 'wb') as f:
            dill.dump(fr_final, f)  # Salva o modelo completo, incluindo o regressor e o pipeline
            print(f"Modelo salvo em {model_file}")

        shutil.copy(config_file, config_file_copy)
        
        end = time.perf_counter()
        training_time = end - start

        with open(os.path.join(save_path, 'training_time.txt'), "w") as f:
            f.write(str(training_time))
    
    else:
        start = time.perf_counter()
        create_clean_directory(config['save_results'])

        metrics_csv = os.path.join(save_path,'metrics.csv')
        model_file = os.path.join(save_path,'fr_model.pkl')
        config_file_copy = os.path.join(save_path,'config.yaml')

        regressor = eval(params['regressor'])

        surrogate = Pipeline([('regressor', regressor)])

        model_builder = eval(config['fr_model_builder'])
        fr = model_builder(X_rom=params['X_rom'], y_rom=params['y_rom'], surrogate=surrogate, surrogate_kwargs={})

        fr.fit(X=training_X, y=training_y)

        prediction = fr.predict(test_X)
        ground_truth = test_y

        metrics_dict = params['metrics']

        metrics = {}
        for key, value in metrics_dict.items():
            metrics[key] = eval(value)(ground_truth, prediction)

        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(metrics_csv)
        
        # Salvando o modelo com dill
        with open(model_file, 'wb') as f:
            dill.dump(fr, f)  # Salva o modelo completo, incluindo o regressor e o pipeline
            print(f"Modelo salvo em {model_file}")

        shutil.copy(config_file, config_file_copy)

        end = time.perf_counter()
        training_time = end - start

        with open(os.path.join(save_path, 'training_time.txt'), "w") as f:
            f.write(str(training_time))


    # Marcar o tempo de fim da execução
    end_time = time.time()

    # Calcular e exibir o tempo total de execução
    elapsed_time = end_time - start_time
    print(f'Tempo de execução: {elapsed_time:.2f} segundos')
    
    return final_metrics, fr_final

if __name__ == "__main__":
    app()