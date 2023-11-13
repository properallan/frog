import pandas as pd
from pathlib import Path

from skopt.space import Space

class DoEGenerator:
    """Design of Experiments class."""
    def __init__(self, 
        variables : dict, 
        sampler: callable,
        n_samples: int,):
        """Initialize the DoE class."""
        self.space = self.gen_space(variables)
        self.df = self.gen_doe(variables, sampler, self.space, n_samples)

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
        

    def run(self):
        for i in range(len(self.df)):
            self.run_design_point(i)
        if self.log_file is not None:
            self.save(self.log_file)

    def run_design_point(self, design_point: int):
        """Run the design point."""
        config = {**self.df.iloc[design_point].to_dict(), **self.other_variables}
        config['working_dir'] = Path(config['working_dir']) / str(design_point)
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

    @property
    def design_points(self):
        return len(self.df.index)