import pandas as pd
from pathlib import Path
from skopt.space import Space
import ray

from importlib import import_module
import os
import sys

sys.path.append(os.getcwd())

def load_func(dotpath : str):
    """ load function in module.  function is right-most segment """
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)

class DoEGenerator:
    """Design of Experiments class."""
    def __init__(self, 
        variables : dict, 
        sampler: callable,
        n_samples: int,
        other_variables: dict = None):
        """Initialize the DoE class."""
        self.space = self.gen_space(variables)
        self.df = self.gen_doe(variables, sampler, self.space, n_samples)

        if other_variables is not None:
            self.fdf = self.df.assign(**other_variables)
            for i in self.df.index:
                self.fdf.loc[i,'working_dir'] = Path(self.fdf.loc[i,'working_dir']) / str(i)


    def gen_space(self, variables: dict) -> object:
        """Generate the space."""
        return Space(list(variables.values()))

    def gen_doe(self, 
        variables: dict, 
        sampler: object,
        space: object, 
        n_samples) -> pd.DataFrame:
        """Generate the design of experiments."""
        samples = sampler.generate(space.dimensions, n_samples)
        doe_df =  pd.DataFrame(samples, columns=variables.keys())
        doe_df.index.rename('design_point', inplace=True)
        
        return doe_df
    
    def run_design_point(self, design_point: int):
        """Run the design point."""
        self.function(self.doe[design_point])

    def save(self, file: str):
        """Save the design of experiments."""
        file = Path(file)
        if not file.parent.exists():
            file.parent.mkdir(parents=True)
        self.df.to_csv(file)

    @property
    def design_points(self):
        return len(self.df.index)
    
class DoEPreProcessor:
    def __init__(self, 
        file: str, 
        other_variables: dict):
        self.file = Path(file)
        self.df = pd.read_csv(file, index_col='design_point')
        self.df['status'] = 'pending'
        self.other_variables = other_variables
        self.df = self.df.assign(**other_variables)
        for i in self.df.index:
            self.df.loc[i,'working_dir'] = Path(self.df.loc[i,'working_dir']) / str(i)
    
    def save(self, file: str):
        """Save the design of experiments."""
        file = Path(file)
        if not file.parent.exists():
            file.parent.mkdir(parents=True)
        self.df.to_csv(file)

    
class DoEProcessor:
    def __init__(self,
        doe_file: str):
        self.df = pd.read_csv(doe_file, index_col='design_point')
        self.function = load_func(self.df['function'][0])

    def run(self):
        @ray.remote(
            num_cpus=1, 
            max_calls=16, 
            scheduling_strategy='SPREAD')
        def remote_function(*args, **kwargs):
            return self.function(*args, **kwargs)

        queue = []
        for design_point in range(len(self.df)):
            config = self.df.iloc[design_point].to_dict()
            queue.append( remote_function.remote(config))
        
        results = ray.get(queue)
        for design_point in range(len(self.df)):
            if results[design_point] == True:
                self.df.loc[design_point,'status'] = 'success'
            else:
                self.df.loc[design_point,'status'] = 'failed'

    def run_design_point(self, design_point: int):
        """Run the design point."""
        config = self.df.iloc[design_point].to_dict()
        result = self.function(config)
        if result == True:
            self.df.loc[design_point,'status'] = 'success'
        else:
            self.df.loc[design_point,'status'] = 'failed'

    def save(self, file: str):
        """Save the design of experiments."""
        file = Path(file)
        if not file.parent.exists():
            file.parent.mkdir(parents=True)
        self.df.to_csv(file)

    
class DoERunner:
    def __init__(self, 
        file: str, 
        function: callable,
        other_variables: dict,
        log_file: str = None):
        self.file = Path(file)
        self.log_file = log_file
        self.df = pd.read_csv(file, index_col='design_point')
        self.df['status'] = 'pending'
        self.function = function
        self.other_variables = other_variables
    
    #def run(self):
    #    for i in range(len(self.df)):
    #        self.run_design_point(i)
    #    if self.log_file is not None:
    #        self.save(self.log_file)

    def run(self):
        @ray.remote(
            num_cpus=1, 
            max_calls=16, 
            scheduling_strategy='SPREAD')
        def remote_function(*args, **kwargs):
            return self.function(*args, **kwargs)

        queue = []
        for design_point in range(len(self.df)):
            config = {**self.df.iloc[design_point].to_dict(), **self.other_variables}
            config['working_dir'] = Path(config['working_dir']) / str(design_point)
            queue.append( remote_function.remote(config))
        
        results = ray.get(queue)
        for design_point in range(len(self.df)):
            if results[design_point] == True:
                self.df.loc[design_point,'status'] = 'success'
            else:
                self.df.loc[design_point,'status'] = 'failed'


    def run_design_point(self, design_point: int):
        """Run the design point."""
        config = {**self.df.iloc[design_point].to_dict(), **self.other_variables}
        config['working_dir'] = Path(config['working_dir']) / str(design_point)
        result = self.function(config)
        if result == True:
            self.df.loc[design_point,'status'] = 'success'
        else:
            self.df.loc[design_point,'status'] = 'failed'

    @ray.remote
    def run_design_point_parallel(self, design_point: int):
        """Run the design point."""
        config = {**self.df.iloc[design_point].to_dict(), **self.other_variables}
        config['working_dir'] = Path(config['working_dir']) / str(design_point)
        result = self.remote_function.remote(config)
        if result == True:
            self.df.loc[design_point,'status'] = 'success'
        else:
            self.df.loc[design_point,'status'] = 'failed'

    def save(self, file: str):
        """Save the design of experiments."""
        file = Path(file)
        if not file.parent.exists():
            file.parent.mkdir(parents=True)
        self.df.to_csv(file)

    @property
    def design_points(self):
        return len(self.df.index)
