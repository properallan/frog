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

    import dill
    import os
    import pandas as pd
    from frog.utils import create_clean_directory
    import shutil
    import numpy as np

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


    # Filter variables
    training_X = training_X[params['LF_VARIABLES']]
    training_y = training_y[params['HF_VARIABLES']]

    regressor = eval(params['regressor'])

    surrogate = Pipeline([('regressor', regressor)])

    # KFold Cross Validation Setup
    if 'kfold_cross_validation' in config.keys():
        from sklearn.model_selection import KFold
        kfold = KFold(**config['kfold_cross_validation'])  # Exemplo com 5 splits

        fold_metrics = []  # Lista para armazenar as métricas de cada fold
        fold_models = []  # Lista para armazenar os modelos de cada fold

        for fold_num, (train_index, val_index) in enumerate(kfold.split(training_X)):
            X_train, X_val = training_X[train_index], training_X[val_index]
            y_train, y_val = training_y[train_index], training_y[val_index]

            # Criar e treinar o modelo
            model_builder = eval(config['fr_model_builder'])
            fr = model_builder(X_rom=params['X_rom'], y_rom=params['y_rom'], surrogate=surrogate, surrogate_kwargs={})

            fr.fit(X=X_train, y=y_train)

            # Fazer a previsão
            prediction = fr.predict(X_val)

            # Calcular as métricas (exemplo com MSE)
            metrics_dict = params['metrics']
            metrics = {}
            for key, value in metrics_dict.items():
                metrics[key] = eval(value)(y_val, prediction)

            fold_metrics.append(metrics)

            # Salvar o modelo de cada fold
            fold_model_file = os.path.join(config['save_results'], f"fold_{fold_num + 1}", f"fr_model.pkl")
            with open(fold_model_file, 'wb') as f:
                dill.dump(fr, f)
                print(f"Modelo do fold {fold_num + 1} salvo em {fold_model_file}")

            fold_models.append(fold_model_file)

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
            
        # Marcar o tempo de fim da execução
        end_time = time.time()

        # Calcular e exibir o tempo total de execução
        elapsed_time = end_time - start_time
        print(f'Tempo de execução: {elapsed_time:.2f} segundos')

        # Exibir os resultados das métricas médias
        print("Média das métricas após K-Fold Cross Validation:")
        print(avg_metrics)

        

        # Salvar as métricas de cada fold e as métricas médias
        if 'save_results' in config.keys():
            create_clean_directory(config['save_results'])

            metrics_csv = os.path.join(config['save_results'], 'metrics.csv')
            metrics_folds_csv = os.path.join(config['save_results'], 'metrics_folds.csv')
            final_metrics_csv = os.path.join(config['save_results'], 'final_metrics.csv')
            model_file = os.path.join(config['save_results'], 'final_model.pkl')
            config_file_copy = os.path.join(config['save_results'], 'config.yaml')

            final_metrics_df = pd.DataFrame([final_metrics])
            final_metrics_df.to_csv(final_metrics_csv, index=False)

            # Salvar as métricas de cada fold
            fold_metrics_df = pd.DataFrame(fold_metrics)
            fold_metrics_df.to_csv(metrics_folds_csv, index=False)

            # Salvar as métricas médias
            avg_metrics_df = pd.DataFrame([avg_metrics])
            avg_metrics_df.to_csv(metrics_csv, index=False)


            # Salvar o modelo final
            with open(model_file, 'wb') as f:
                dill.dump(fr_final, f)
                print(f"Modelo final salvo em {model_file}")

            # Copiar o arquivo de configuração
            shutil.copy(config_file, config_file_copy)

            metrics = ave_metrics
            fr = fr_final
    else:
        model_builder = eval(config['fr_model_builder'])
        fr = model_builder(X_rom=params['X_rom'], y_rom=params['y_rom'], surrogate=surrogate, surrogate_kwargs={})

        fr.fit(X=training_X, y=training_y)

        prediction = fr.predict(test_X)
        ground_truth = test_y

        metrics_dict = params['metrics']

        metrics = {}
        for key, value in metrics_dict.items():
            metrics[key] = eval(value)(ground_truth, prediction)

        # Marcar o tempo de fim da execução
        end_time = time.time()

        # Calcular e exibir o tempo total de execução
        elapsed_time = end_time - start_time
        print(f'Tempo de execução: {elapsed_time:.2f} segundos')

        # Exibir os resultados das métricas
        print(metrics)

        if 'save_results' in config.keys():
            import dill
            import os
            import pandas as pd
            from frog.utils import create_clean_directory
            import shutil

            create_clean_directory(config['save_results'])

            metrics_csv = os.path.join(config['save_results'],'metrics.csv')
            model_file = os.path.join(config['save_results'],'fr_model.pkl')
            config_file_copy = os.path.join(config['save_results'],'config.yaml')
            
            metrics_df = pd.DataFrame(metrics, index=[0])
            metrics_df.to_csv(metrics_csv)
            
            # Salvando o modelo com dill
            with open(model_file, 'wb') as f:
                dill.dump(fr, f)  # Salva o modelo completo, incluindo o regressor e o pipeline
                print(f"Modelo salvo em {model_file}")

            shutil.copy(config_file, config_file_copy)

    return metrics, fr

if __name__ == "__main__":
    app()
