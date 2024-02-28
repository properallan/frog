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
import yaml
from frog.normalization import RowScaler
from frog.transformers import IdentityTransformer

def eval_dict(d):
    from frog.metrics import NRMSE, R2, MAPE, MAXPE

    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = eval_dict(value)
        else:
            if value != 'min':
                d[key] = eval(value)
            else:
                d[key] = value
    return d