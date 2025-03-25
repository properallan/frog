import yaml
from pathlib import Path
from frog.doe import DoEGenerator, DoEPreProcessor, DoEProcessor
from skopt.sampler import Lhs
from importlib import import_module
import os
import sys
import typer
from typing import Annotated

app = typer.Typer()

sys.path.append(os.getcwd())

def load_func(dotpath : str):
    """ load function in module.  function is right-most segment """
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)

@app.command()
def generate(
    config_file: Annotated[str, typer.Argument(help='YAML configuration file to generate the design of experiments.')]):
    """Generate the design of experiments."""

    with open(config_file) as f:
        config = yaml.safe_load(f)

    wd = os.getcwd()
    os.chdir(Path(config_file).resolve().parent)
    
    if 'random_state' not in config:
        config['random_state'] = None
        
    doe_gen = DoEGenerator(
        variables=config['variables'],
        sampler = eval(config['sampler'])(**config['sampler_args']),
        n_samples = config['n_samples'],
        random_state = config['random_state']
    )

    doe_gen.save(config['output_file'])
    os.chdir(wd)
    
    return doe_gen

@app.command()
def process(
    doe_file: Annotated[str, typer.Argument(help='CSV file containing the design of experiments.')], 
    log_file: Annotated[str, typer.Option(help='Optional log file with processing results.')]=None):
    """Run the design of experiments."""

    doe_run = DoEProcessor(
        doe_file=Path(doe_file))

    wd = os.getcwd()
    os.chdir(Path(doe_file).resolve().parent)
    doe_run.run()
    os.chdir(wd)

    if log_file is None:
        log_file = Path(doe_file).with_suffix('.log.csv').__str__()
    doe_run.save(log_file)



@app.command()
def preprocess(
    config_file: Annotated[str, typer.Argument(help='YAML configuration file to generate the design of experiments full file with other variables.')]):
    """Configure the design of experiments runner."""
    with open(config_file) as f:
        config = yaml.safe_load(f)

    os.chdir(Path(config_file).resolve().parent)

    doe_run = DoEPreProcessor(
        file=Path(config['doe_file']),
        other_variables = config['other_variables'],
    )

    doe_run.save(config['output_file'])
    return doe_run

@app.command()
def run(config_file: Annotated[str, typer.Argument(help='YAML configuration file to run the design of experiments.')]):
    """Run the design of experiments."""
    from frog.datahandler import generate as dataset_generate
    import os

    with open(config_file) as f:
        config = yaml.safe_load(f)

    os.chdir(Path(config_file).resolve().parent)
    config_file = Path(config_file).name
    generate(config_file = config_file)
    os.chdir(Path(config_file).resolve().parent)

    preprocess(config_file = config['lf_config_file'])
    os.chdir(Path(config_file).resolve().parent)
    with open(config['lf_config_file']) as f:
        lf_config = yaml.safe_load(f)
    process(doe_file = lf_config['output_file'])
    os.chdir(Path(config_file).resolve().parent)

    preprocess(config_file = config['hf_config_file'])
    os.chdir(Path(config_file).resolve().parent)
    with open(config['hf_config_file']) as f:
        hf_config = yaml.safe_load(f)
    process(doe_file = hf_config['output_file'])
    os.chdir(Path(config_file).resolve().parent)

    dataset_generate(config['dataset_config_file'])
    os.chdir(Path(config_file).resolve().parent)

    
if __name__ == "__main__":
    app()