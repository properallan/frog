def get_best_config(hpo_csv_file, metric='R2', ascending=False, n_trials=1):
    import pandas as pd

    hpo = pd.read_csv(hpo_csv_file)
    hpo.sort_values(by=metric, inplace=True, ascending=ascending)

    best_configs = []
    best_metrics = []
    best_trials = []

    for i in range(n_trials):
        params = hpo.iloc[i].to_dict()

        config = {}
        for key, value in params.items():
            if 'config' in key:
                config[key.split('/')[-1]] = value
        
        best_configs.append(config)
        best_metrics.append({metric: params[metric]})
        best_trials.append(params)

    return best_configs, best_metrics, best_trials