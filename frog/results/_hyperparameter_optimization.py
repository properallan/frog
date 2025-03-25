import pandas as pd
from typing import Union
from pathlib import Path

def get_best_result(
    optimization_results: Union[str, Path],
    metric: str = None,
    ascending: bool = True,
    filter: str = None,
):
    if not isinstance(optimization_results, Path) and isinstance(optimization_results, str):
        optimization_results = Path(optimization_results)

    results_df = pd.read_csv(optimization_results)

    if filter is not None:
        results_df = results_df.query(filter)

    if metric is not None:
        results_df = results_df.sort_values(by=metric, ascending=ascending)

    return results_df