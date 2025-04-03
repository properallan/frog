import typer
from typing import Annotated
import sys
import os

app = typer.Typer()

sys.path.append(os.getcwd())


import os
import shutil

def create_clean_directory(dir_path, overwrite=False):
    """
    Cria um diretório. Se já existir, remove todo o conteúdo e recria do zero.

    Args:
        dir_path (str): Caminho do diretório a ser criado/limpo.
    """
    if os.path.exists(dir_path) and overwrite:
        shutil.rmtree(dir_path)  # Remove o diretório e todo o conteúdo
    os.makedirs(dir_path, exist_ok=True)  # Recria o diretório vazio


@app.command()
def optimize(
    config_file: Annotated[str, typer.Argument(help='YAML configuration file to run the hyperparameter optimization.')],
    restore: bool = typer.Option(False, "-r", "--restore", help="Restore hyperparameter optimization")):
    from frog.optimization import HyperOpt
    from frog.flow_reconstruction import FlowReconstruction
    from frog.metrics import NRMSE, R2, MAPE, MAXPE, MSE, MAE
    from pathlib import Path
    import numpy as np
    from hyperopt import hp
    from hyperopt.pyll import scope
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search.bayesopt import BayesOptSearch
    from ray.tune.search.hebo import HEBOSearch
    import yaml
    import tensorflow as tf
    from frog.utils import load_func, eval_dict

    config = yaml.safe_load(open(config_file))

    import os
    #os.chdir(Path(config_file).resolve().parent)
   

    search_space = eval_dict(config['search_space'])
    
    if 'TRAINING_X' in config['other_params']:
        config['other_params']['TRAINING_X'] = Path(config['other_params']['TRAINING_X']).resolve().__str__()
    if 'TRAINING_y' in config['other_params']:
        config['other_params']['TRAINING_y'] = Path(config['other_params']['TRAINING_y']).resolve().__str__()
    if 'TEST_X' in config['other_params']:
        config['other_params']['TEST_X'] = Path(config['other_params']['TEST_X']).resolve().__str__()
    if 'TEST_y' in config['other_params']:
        config['other_params']['TEST_y'] = Path(config['other_params']['TEST_y']).resolve().__str__()
    if 'VALIDATION_X' in config['other_params']:
        config['other_params']['VALIDATION_X'] = Path(config['other_params']['VALIDATION_X']).resolve().__str__()
    if 'VALIDATION_y' in config['other_params']:
        config['other_params']['VALIDATION_y'] = Path(config['other_params']['VALIDATION_y']).resolve().__str__()

    other_params = config['other_params']

    model_builder = eval(config['model_builder'])

    optimize_kwargs = config['optimize']
    #optimize_kwargs['hyperopt_path'] = (Path(config['optimize']['hyperopt_path']).resolve() / Path(config_file).stem).__str__()
    optimize_kwargs['hyperopt_path'] = Path(config['optimize']['hyperopt_path']).resolve().__str__()

    #optimize_kwargs['search_algorithm'] = eval(optimize_kwargs['search_algorithm'])
    optimize_kwargs['resources'] = eval(optimize_kwargs['resources'])

    other_params['model_builder'] = config['model_builder']
    other_params['metrics'] = config['metrics']

    objective_function = load_func(config['objective_function'])

    other_params['early_stopping'] = config['early_stopping']

    hyperopt = HyperOpt(
        objective_function=objective_function, 
        search_space=search_space, 
        other_params=other_params,
        model_builder=model_builder,
    )

    if restore:
        hyperopt.restore(
            objective_function=objective_function,
            other_params=other_params,
            experiment_name=optimize_kwargs['hyperopt_name'],
            hyperopt_path=optimize_kwargs['hyperopt_path'],
            resources=optimize_kwargs['resources']
        )
    else:
        hyperopt.optimize(
            objective_function=objective_function, 
            search_space=search_space, 
            other_params=other_params, 
            experiment_name=optimize_kwargs['hyperopt_name'],
            **optimize_kwargs
        )

if __name__ == "__main__":
    app()