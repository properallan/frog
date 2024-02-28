# Hyperparemeters optimization
from frog.flow_reconstruction import FRBuilder, FRNNBuilder, FRKrigingBuilder
from ray import train, tune
from ray.train import RunConfig#, CheckpointConfig
from ray.tune.search.hyperopt import HyperOptSearch
from frog.metrics import NRMSE, R2, MAPE
from pathlib import Path


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
                storage_path=Path(study_path).resolve().__str__(),
                name=study_name,
                #checkpoint_config=CheckpointConfig(),
            ),
            tune_config=tune.TuneConfig(num_samples=1),
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
    
    def optimize(self, 
        objective_function=None, 
        search_space=None, 
        other_params={}, 
        search_algorithm=HyperOptSearch, 
        num_samples=1000, 
        metric='nrmse', 
        mode='min',
        hyperopt_path=None,
        resources=None, **kwargs):
        if hyperopt_path is None:
            hyperopt_path = Path(self.other_params['PATH']) / 'hyperopt'
        if objective_function is None:
            objective_function = self.objective_function
        if other_params == {}:
            other_params = self.other_params
        if search_space is None:
            search_space = self.search_space

        with_parameters = tune.with_parameters(
            objective_function, 
            other_params=other_params, 
        )
        
        #with_resources = tune.with_resources(with_parameters, 
        #                        {
        #                            "cpu": 1,
        #                            #"gpu": 0, 
        #                            #"memory": (128/56)*10**9
        #                        })
        
        with_resources = tune.with_resources(
            with_parameters, resources)

        search_algo = search_algorithm(search_space, metric=metric, mode=mode)

        tuner = tune.Tuner(
            with_resources,
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                search_alg=search_algo,
            ),
            run_config=RunConfig(
                storage_path=Path(hyperopt_path).parent,
                name=Path(hyperopt_path).stem,
            ),
        )

        results = tuner.fit()

        #print(results.get_best_result(metric="nrmse", mode="min").config)
        
        self.results = results
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