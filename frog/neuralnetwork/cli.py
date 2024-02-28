import typer
from typing import Annotated
import sys
import os

app = typer.Typer()

sys.path.append(os.getcwd())

@app.command()
def optimize(config_file: Annotated[str, typer.Argument(help='YAML configuration file to run the hyperparameter optimization.')]):
    from frog.optimization import HyperOpt
    from frog.flow_reconstruction import FlowReconstruction
    from frog.metrics import NRMSE, R2, MAPE, MAXPE, MSE, MAE
    from pathlib import Path
    import numpy as np
    from hyperopt import hp
    from hyperopt.pyll import scope
    from ray.tune.search.hyperopt import HyperOptSearch
    import yaml
    import tensorflow as tf
    from frog.utils import load_func, eval_dict

    config = yaml.safe_load(open(config_file))

    import os
    os.chdir(Path(config_file).resolve().parent)

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
    optimize_kwargs['hyperopt_path'] = (Path(config['optimize']['hyperopt_path']).resolve() / Path(config_file).stem).__str__()

    optimize_kwargs['search_algorithm'] = eval(optimize_kwargs['search_algorithm'])
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

    hyperopt.optimize(
        objective_function=objective_function, 
        search_space=search_space, 
        other_params=other_params, 
        **optimize_kwargs
    )

if __name__ == "__main__":
    app()