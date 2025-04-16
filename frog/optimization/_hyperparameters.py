# Hyperparemeters optimization
from frog.flow_reconstruction import FRBuilder, FRNNBuilder, FRKrigingBuilder
from ray import train, tune
from ray.train import RunConfig#, CheckpointConfig
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.hebo import HEBOSearch
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from frog.schedulers import CustomHyperBandForBOHB
from frog.metrics import NRMSE, R2, MAPE, MAE, MAXPE, MSE
from pathlib import Path
from ray.train import CheckpointConfig, SyncConfig

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

class GridSearch:
    def __init__(self,
        objective_function : callable, 
        search_space : dict,
        other_params : dict,
        model_builder : FRBuilder,
        study_name, 
        study_path) -> None:
        
        self.objective_function = objective_function
        self.search_space = search_space
        self.other_params = other_params
        self.model_builder = model_builder
        self.study_name = study_name
        self.study_path = study_path

    def get_best_model(self, metric='nrmse', mode='min'):
        config = self.results.get_best_result(metric=metric, mode=mode).config
        self.fr = self.model_builder(**{**config, **self.other_params})
        self.fr.fit()
        return self.fr

    def get_best_parameters(self, metric='nrmse', mode='min'):
        return self.results.get_best_result(metric=metric, mode=mode).config
    
    def optimize(self,
        objective_function=None,
        search_space=None,
        other_params={},
        study_path: Path=None,
        study_name=None,
        resources={'memory':8 * 1024 * 1024 * 1024, 'cpu': 1},
        model_builder=None,
        **kwargs):
        if objective_function is None:
            objective_function = self.objective_function
        if search_space is None:
            search_space = self.search_space
        if other_params == {}:
            other_params = self.other_params
        if study_path is None:
            study_path = self.study_path
        if study_name is None:
            study_name = self.study_name
        if model_builder is None:
            model_builder = self.model_builder
        

        trainable_wr = tune.with_resources(
            trainable=objective_function, 
            resources=resources)
        
        trainable_wp = tune.with_parameters(
            trainable_wr, 
            other_params=other_params,
            model_builder=model_builder)

        tuner = tune.Tuner(
            trainable=trainable_wp,
            param_space=search_space,
            run_config=train.RunConfig(
                storage_path=Path(study_path).parent.resolve(),
                name=study_name,
                #checkpoint_config=CheckpointConfig(),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=None,
                    checkpoint_frequency=0
                ),
                log_to_file=True,
                #local_dir=Path(study_path).parent,
            ),
            #tune_config=tune.TuneConfig(num_samples=1),
        )

        results = tuner.fit()

        results.get_dataframe().to_csv((Path(study_path) / study_name).with_suffix('.csv').__str__())
                
        
class HyperOpt:
    def __init__(self,
        objective_function : callable, 
        search_space : dict,
        other_params : dict,
        model_builder : FRBuilder) -> None:
        
        self.objective_function = objective_function
        self.search_space = search_space
        self.other_params = other_params
        self.model_builder = model_builder

    def plot(self, x, y):
        # x is a list of hyperparameters
        # y is a list of objective values
        pass
    
    def get_best_model(self, metric='nrmse', mode='min'):
        config = self.results.get_best_result(metric=metric, mode=mode).config
        self.fr = self.model_builder(**{**config, **self.other_params})
        self.fr.fit()
        return self.fr
    
    def get_best_parameters(self, metric='nrmse', mode='min'):
        return self.results.get_best_result(metric=metric, mode=mode).config

    def restore(self,
        objective_function=None,
        other_params=None,
        resources=None,
        hyperopt_path=None,
        experiment_name=None,
        resume_errored=False,
        restart_errored=False):

        import os
        #os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = Path(hyperopt_path).resolve().__str__()

        experiment_path = (Path(hyperopt_path).resolve() / Path(experiment_name)).__str__()

        with_parameters = tune.with_parameters(
            objective_function, 
            other_params=other_params, 
        )
        
        with_resources = tune.with_resources(
            with_parameters, resources)

        if os.path.exists(experiment_path):
            tuner = tune.Tuner.restore(
            path=experiment_path,
            trainable=with_resources,
            resume_errored=resume_errored,
            restart_errored=restart_errored
        )
            
        results = tuner.fit()

        self.results = results

        study_path = Path(hyperopt_path)
        study_name = Path(experiment_name)
        results.get_dataframe().to_csv((Path(study_path) / study_name / 'results').with_suffix('.csv').__str__())

        return results    
    
    def optimize(self, 
        objective_function=None, 
        search_space=None, 
        other_params={}, 
        search_algorithm='HyperOptSearch', 
        search_scheduler='CustomHyperBandForBOHB',
        search_algorithm_kwargs={},
        num_samples=1000, 
        metric='nrmse', 
        mode='min',
        hyperopt_path=None,
        experiment_name=None,
        resources=None, **kwargs):
        # if hyperopt_path is None:
        #     hyperopt_path = Path(self.other_params['PATH']) / 'hyperopt'
        # if objective_function is None:
        #     objective_function = self.objective_function
        # if other_params == {}:
        #     other_params = self.other_params
        # if search_space is None:
        #     search_space = self.search_space

        
        import os
        from pathlib import Path
        #os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = Path(hyperopt_path).resolve().__str__()

        # Exemplo de uso
        
        experiment_path = (Path(hyperopt_path).resolve() / Path(experiment_name)).__str__()
        create_clean_directory(experiment_path, overwrite=True)

        #os.makedirs(optimize_kwargs['hyperopt_path'], exist_ok=True)
        #os.chdir(hyperopt_path)

        with_parameters = tune.with_parameters(
            objective_function, 
            other_params=other_params, 
        )
        
        with_resources = tune.with_resources(
            with_parameters, resources)

        if search_algorithm == 'TuneBOHB':
            from frog.utils import eval_dict
            search_algo = eval(search_algorithm)()
            #print('TUNEBOHBBBBBBBBBBBBBBBBBBB')
            #print(search_algorithm_kwargs)
            #print(search_algorithm_kwargs)
            scheduler = eval(search_scheduler)(
                **search_algorithm_kwargs)
            #print(scheduler._eta)
            tuner = tune.Tuner(
                with_resources,
                tune_config=tune.TuneConfig(
                    metric=metric,
                    mode=mode,
                    search_alg=search_algo,
                    scheduler=scheduler,
                    num_samples=num_samples,
                    reuse_actors=True,
                ),
                run_config=train.RunConfig(
                    #storage_path=Path(hyperopt_path).parent,
                    storage_path=Path(hyperopt_path).resolve(),
                    name=experiment_name,
                    #name=Path(hyperopt_path).stem,
                    #stop={"time_total_s": 60},
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=1,
                        checkpoint_frequency=0,
                        #checkpoint_at_end=True,
                    ),
                    log_to_file=True,
                    #local_dir=Path(hyperopt_path).parent,
                ),
                param_space=search_space,
            )
        else:

            #search_algo = eval(search_algorithm)(metric=metric, mode=mode)
            search_algo = eval(search_algorithm)(search_space, metric=metric, mode=mode)

            tuner = tune.Tuner(
                with_resources,
                tune_config=tune.TuneConfig(
                    num_samples=num_samples,
                    search_alg=search_algo,
                    reuse_actors=True,
                ),
                run_config=RunConfig(
                    #storage_path=Path(hyperopt_path).parent,
                    storage_path=Path(hyperopt_path).resolve(),
                    name=experiment_name,
                    #name=Path(hyperopt_path).stem,
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=1,
                        checkpoint_frequency=0,
                       #checkpoint_at_end=True,
                    ),
                    log_to_file=True,
                    #local_dir=Path(hyperopt_path).parent,
                ),
                #param_space=search_space
            )

        results = tuner.fit()

        #print(results.get_best_result(metric="nrmse", mode="min").config)
        
        self.results = results

        study_path = Path(hyperopt_path)
        study_name = Path(experiment_name)
        results.get_dataframe().to_csv((Path(study_path) / study_name / 'results').with_suffix('.csv').__str__())

        return results

    def objective(self, config:dict, other_params:dict={}) -> dict:
        """
        Objective function for hyperparameter optimization.
        
        Parameters
        ----------
        config : dict
            Dictionary of hyperparameters.
        other_params : dict, optional
            Dictionary of other parameters, by default {}.

        Returns
        -------
        dict
            Dictionary of metrics.
        """

        fr = self.model_builder(**{**config, **other_params})
        fr.fit(X=fr.snapshots_X_train, y=fr.snapshots_y_train)
        
        prediction = fr.predict(fr.snapshots_X_test)
        ground_truth = fr.snapshots_y_test

        nrmse = NRMSE(ground_truth, prediction)
        r2 = R2(ground_truth, prediction)

        return {'nrmse' : nrmse, 'r2' : r2}

def objective(config:dict, other_params:dict={}) -> dict:
    """
    Objective function for hyperparameter optimization.
    
    Parameters
    ----------
    config : dict
        Dictionary of hyperparameters.
    other_params : dict, optional
        Dictionary of other parameters, by default {}.

    Returns
    -------
    dict
        Dictionary of metrics.
    """

    fr = FRKrigingBuilder(**{**config, **other_params})
    fr.fit(X=fr.snapshots_X_train, y=fr.snapshots_y_train)
    
    prediction = fr.predict(fr.snapshots_X_test)
    ground_truth = fr.snapshots_y_test

    nrmse = NRMSE(ground_truth, prediction)
    r2 = R2(ground_truth, prediction)

    return {'nrmse' : nrmse, 'r2' : r2}