import pandas as pd
from pathlib import Path
from skopt.space import Space
import ray

from importlib import import_module
import os
import sys

from copy import copy
import numpy as np

import xarray as xr
from typing import Union
from collections.abc import Iterable
import pyvista as pv

from loguru import logger

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
    
class AbstractProcessor:
    def save(self, file: str):
        """Save the design of experiments."""
        file = Path(file)
        if not file.parent.exists():
            file.parent.mkdir(parents=True)
        self.df.to_csv(file)

class DoEPreProcessor(AbstractProcessor):
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
    
class DoEProcessor(AbstractProcessor):
    def __init__(self,
        doe_file: str):
        self.df = pd.read_csv(doe_file, index_col='design_point')
        self.function = load_func(self.df['function'][0])

    def run(self):
        #rcontext = ray.init(include_dashboard=True)
        #print(rcontext.dashboard_url)
        #input('press any key to continue')
        
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

def get_residue_2D(config, tol=None) -> bool:
        cwd = Path(config['working_dir'])
        if tol is None:
            tol = config['solver_tol']

        residue = pd.read_csv(cwd / 'history.csv', header=0, delimiter=',')
        keys = residue.keys()

        max_residue = -np.inf*-1
        for i in range(1, len(keys)):
            if residue[keys[i]].values[-1] < max_residue:
                max_residue = residue[keys[i]].values[-1]

        return max_residue

def get_residue_1D(config, tol=None, check_residuesssssssss='any') -> bool: 
        if tol is None:
            tol = config['solver_tol']

        maxres = -np.inf*-1
        for residue in ['rese.txt', 'resrho.txt', 'resrhou.txt']:
            last_residue = np.loadtxt(Path(config['working_dir']) / 'outputs' / residue)[-1]
            if last_residue < maxres:
                maxres = last_residue

        return maxres

def check_convergence(residue, tol):
    converged = False
    if residue <= tol:
        converged = True
    return converged

class ConvergenceChecker:
    def __init__(self, residue_function : callable):
        self.residue_function = residue_function

    def check(self, config: dict,  tol=None):
        if tol is None:
            tol = config['solver_tol']
        residue = self.residue_function(config = config, tol=tol)
        return check_convergence(residue=residue, tol=tol)
    
    def check_dataframe(self, dataframe: Union[pd.DataFrame, str], tol=None):
        if isinstance(dataframe, str):
            dataframe = pd.read_csv(dataframe)
        for i, row in dataframe.iterrows():
            dataframe.loc[i, 'converged'] =  self.check(row.to_dict(), tol)
        return dataframe

    def get_converged(self, dataframe: Union[pd.DataFrame, str], tol=None):
        if tol is None:
            tol = self.tol
        converged = self.check_dataframe( dataframe=dataframe, tol=tol)
        return converged[converged['converged']== True] 
    
class DoePostProcessor:
    def __init__(self, 
        doe_file: str=None, 
        dataframe: pd.DataFrame=None,
        filter: dict={},
        snapshot_function: callable=None,
        #convergence_checker: ConvergenceChecker
        ):
        
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(doe_file, index_col='design_point')

        self.df = self.filter(filter=filter)

        self.snapshot_function = snapshot_function

    def filter(self, filter:dict={}) -> pd.DataFrame:
        for key, value in filter.items():
            self.df = self.df[self.df[key]==value]

        return self.df
    
    def get_snapshot_and_index(self, design_point: int, *args, **kwargs):
        snapshot, idx_dict =  self.snapshot_function(self.df.loc[design_point, :], *args, **kwargs)
        self.idx_dict = idx_dict
        return snapshot, idx_dict
        
    def get_snapshot(self, design_point: int, *args, **kwargs):
        snapshot, idx_dict =  self.get_snapshots_and_indexes(design_point, *args, **kwargs)
        return snapshot
    
    def get_design_points(self):
        return self.df['design_point']
            
    def get_snapshots_and_indexes(self, *args, **kwargs):
        snapshots = None
        for i, row in self.df.iterrows():
            #snapshot, idx_dict = self.snapshot_function(self.df.loc[i, :], *args, **kwargs)
            snapshot, idx_dict = self.get_snapshot_and_index(i, *args, **kwargs)
            if snapshots is None:
                snapshots = snapshot
            else:
                snapshots = np.vstack([snapshots, snapshot])
        self.idx_dict = idx_dict
        return snapshots, idx_dict

    def get_snapshots(self, *args, **kwargs):
        snapshots ,idx_dict = self.get_snapshots_and_indexes(*args, **kwargs)
        return snapshots


    def check_convergence(self):
        pass

    def filter_index(self, index: Iterable):
        self.df = self.df[self.df.index.isin(index)]
        return self.df
        

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


def get_2D_snapshot(conf : dict={}, variables: list = None, dimensionalization={}):
    working_dir = conf['working_dir']
    vtm = pv.read(Path(working_dir) / 'fluid.vtm')
    #vtm[0]['Boundary']['UPPER_WALL']
    #dir(vtm[0]['Internal']['Internal'])

    var_array = np.array([])
    idx_dict = {}
    Internal = vtm[0]['Internal']['Internal']
    if variables is None:
        variables_internal = Internal.array_names
    else:
        variables_internal =  [v for v in variables if v.find('/')<0]
    for variable in variables_internal:
        idx_dict[variable] = slice(var_array.size, var_array.size+Internal[variable][:].flatten().size)
        var_array = np.concatenate([var_array, Internal[variable][:].flatten()])

    other_block = list(np.unique([v.split('/')[0] for v in variables if v.find('/')>0]))

    other_variables_dict = { }
    other_vars_dict = { }
    for ob in other_block:
        other_variables_dict[ob] = [v.split('/')[1] for v in variables if v.find(ob)>=0]
        other_vars_dict[ob] = [v for v in variables if v.find(ob)>=0]
        
    if other_variables_dict.__len__() > 0:
        for ob in other_variables_dict.keys():
            block = vtm[0]['Boundary'][ob]

            for variable, var in zip(other_variables_dict[ob], other_vars_dict[ob]):
                block_flatten= block[variable][2:].flatten()
                idx_dict[var] = slice(var_array.size, var_array.size+block_flatten.size)
                var_array = np.concatenate([var_array, block_flatten])
    
    for var_name, conf_name in dimensionalization.items():
        if var_name in variables:
            var_array[idx_dict[var_name]] = var_array[idx_dict[var_name]]*conf[conf_name]
        
    return var_array, idx_dict

def dict_to_array_and_index(dict, variables=None):
    if variables is None:
        variables = dict.keys()
    var_array = np.array([])
    if len(dict[list(dict.keys())[0]].shape) == 2:
        var_array = np.array([[]]*dict[list(dict.keys())[0]].shape[0])
    idx_dict = {}
    for variable in variables:
        idx_dict[variable] = slice(var_array.shape[-1], var_array.shape[-1]+dict[variable].shape[-1])
        var_array = np.concatenate([var_array, dict[variable]],axis=-1)
    return var_array, idx_dict

def array_and_index_to_dict(array, idx_dict):
    dict = {}
    for variable, idx in idx_dict.items():
        dict[variable] = array[...,idx]
    return dict

def results_to_dict(results_path : str, variables : list = None):    
    array_names = variables
    array_fnames = [(Path(results_path) / var).resolve().with_suffix('.txt') for var in variables]    

    results_dict = {}
    for array_name, array_fname in zip(array_names, array_fnames):
        try:
            result = np.loadtxt(array_fname)
            results_dict[array_name] = result
            #logger.info(f'sucessfuly loaded {array_fname} to numpy array')
        except:
            logger.warning(f'cant load {array_fname} to numpy array')
            pass

    return results_dict

def get_1D_snapshot_full(conf : dict={}, variables: list = None):
    working_dir = conf['working_dir']
    var_dict = results_to_dict(Path(working_dir)/'outputs', variables)
    var_array, idx_dict = dict_to_array_and_index(dict = var_dict)
    return var_array, idx_dict

def get_1D_snapshot(conf : dict={}, variables: list = None):
    var_dict = {}
    
    working_dir = conf['working_dir']

    # separate into scalar and distributed variables
    scalars = ['bc_p0', 'bc_T0', 'bc_Tw','divergence_angle']

    for var in variables:
        if var in scalars:
            var_dict[var] = np.array([conf[var]])
        else:
            tmp_dict = results_to_dict(Path(working_dir)/'outputs', [var])
            var_dict.update(tmp_dict)

    # convert to array and get indexes
    var_array, idx_dict = dict_to_array_and_index(dict = var_dict)

    return var_array, idx_dict



