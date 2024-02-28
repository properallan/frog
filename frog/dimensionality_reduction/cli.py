import typer
from typing import Annotated
from frog.utils import load_func, eval_dict

from frog.optimization import HyperOpt, GridSearch
from frog.flow_reconstruction import FRNNBuilder, FRKrigingBuilder, FRLinearBuilder, FlowReconstruction, FRBuilder
from frog.metrics import NRMSE, R2, MAPE, MAXPE
from pathlib import Path
import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope
from ray.tune.search.hyperopt import HyperOptSearch
import yaml
import tensorflow as tf
from ray import train, tune
import sys
import os
import yaml
from frog.normalization import RowScaler
from frog.transformers import IdentityTransformer

app = typer.Typer()

sys.path.append(os.getcwd())

@app.command()
def study(config_file: Annotated[str, typer.Argument(help='YAML configuration file to run the dimensionality reduction study.')]):
    """Run the dimensionality reduction study."""
    from frog.optimization import HyperOpt, GridSearch

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    os.chdir(Path(config_file).resolve().parent)

    # yaml parser
    objective_function = load_func(config['objective_function'])
    search_space = eval_dict(config['search_space'])
    model_builder = eval(config['fr_model_builder'])
    study_name = Path(config_file).stem
    results_path=config['study_path']
    study_path=Path(config['study_path']).resolve().__str__()

    config['other_params']['TRAINING_X'] = Path(config['other_params']['TRAINING_X']).resolve().__str__()
    config['other_params']['TRAINING_y'] = Path(config['other_params']['TRAINING_y']).resolve().__str__()
    config['other_params']['TEST_X'] = Path(config['other_params']['TEST_X']).resolve().__str__()
    config['other_params']['TEST_y'] = Path(config['other_params']['TEST_y']).resolve().__str__()
    other_params = config['other_params']

    hyperopt = GridSearch(
        objective_function=objective_function, 
        search_space=search_space, 
        other_params=other_params,
        model_builder=model_builder,
        study_name=study_name,
        study_path=study_path,
    )

    optimize_kwargs = config['optimize']
        
    if 'resources' in optimize_kwargs.keys():
        optimize_kwargs['resources'] = eval(optimize_kwargs['resources'])
    
    hyperopt.optimize(
        objective_function=objective_function, 
        search_space=search_space, 
        other_params=other_params, 
        **optimize_kwargs
    )

if __name__ == "__main__":
    app()