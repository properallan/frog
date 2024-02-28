import typer
import sys
import os
import yaml
from typing import Annotated
import numpy as np
#from frog.datahandler._snapshots import generate_dataset
from frog.doe import load_func
from pathlib import Path
app = typer.Typer()

sys.path.append(os.getcwd())

@app.command()
def generate(
    config_file: Annotated[str, typer.Argument(help='YAML configuration file to generate datasets.')]):
    """Generate datasets for flow reconstruction."""


    config_file = Path(config_file).resolve().__str__()
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # generate_dataset(
    #     PATH = config['PATH'],
    #     LF_DOE_FILE = config['LF_DOE_FILE'],
    #     HF_DOE_FILE = config['HF_DOE_FILE'],
    #     LF_VARIABLES = config['LF_VARIABLES'],
    #     HF_VARIABLES = config['HF_VARIABLES'],
    #     HF_TOL_FOR_CONVERGENCE = config['HF_TOL_FOR_CONVERGENCE'],
    #     TEST_RATIO = config['TEST_RATIO'],
    #     VALIDATION_RATIO = config['VALIDATION_RATIO'],)

    load_func(config['function'])(
       **config['function_args'])
    
    return 0