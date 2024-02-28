from frog.doe import DoePostProcessor, ConvergenceChecker, get_1D_snapshot, get_2D_snapshot, get_residue_1D, get_residue_2D
import pandas as pd
from typing import Union
from pathlib import Path
from ._slice_data import split_dataset
import numpy as np
from frog.doe import dict_to_array_and_index, array_and_index_to_dict

def get_snapshot_end_index(npz_file,variables):
    npz = np.load(npz_file, allow_pickle=True)
    snapshots = npz['snapshots']
    index = npz['snapshot_index'].item()
    index_dict = array_and_index_to_dict(snapshots, index)
    snapshots, snapshots_index = dict_to_array_and_index(index_dict, variables)
    return snapshots, snapshots_index

def get_snapshots(LF_DOE_FILE, HF_DOE_FILE, LF_VARIABLES, HF_VARIABLES, HF_TOL_FOR_CONVERGENCE = -8):
    cc2D = ConvergenceChecker(
            residue_function=get_residue_2D
    )

    doe_1D = pd.read_csv(LF_DOE_FILE)

    post_process_1D = DoePostProcessor(
        dataframe = doe_1D,
        snapshot_function = get_1D_snapshot,
    )

    doe_2D = pd.read_csv(HF_DOE_FILE)
    doe_2D = cc2D.check_dataframe(
        dataframe=doe_2D, 
        tol=HF_TOL_FOR_CONVERGENCE,
    )

    post_process_2D = DoePostProcessor(
        dataframe = doe_2D,
        snapshot_function = get_2D_snapshot,
    )

    # filder DOE files to get only converged design points
    post_process_2D.filter( filter = {'converged':True})
    post_process_1D.filter_index(post_process_2D.df['design_point'])


    snapshots_1D = post_process_1D.get_snapshots(
        variables=LF_VARIABLES
    )

    snapshots_2D = post_process_2D.get_snapshots(
        variables = HF_VARIABLES,
        dimensionalization={'Pressure' : 'bc_p0', 'Temperature' : 'bc_T0'}
    )

    return snapshots_1D, snapshots_2D, post_process_1D, post_process_2D


def generate_dataset(
    PATH : Union[str, Path],
    LF_DOE_FILE : Union[str, Path],
    HF_DOE_FILE : Union[str, Path],
    LF_VARIABLES : list,
    HF_VARIABLES : list,
    HF_TOL_FOR_CONVERGENCE : float,
    TEST_RATIO : float,
    VALIDATION_RATIO : float):

    def get_snapshot_end_index(npz_file,variables):
        npz = np.load(npz_file, allow_pickle=True)
        snapshots = npz['snapshots']
        index = npz['snapshot_index'].item()
        index_dict = array_and_index_to_dict(snapshots, index)
        snapshots, snapshots_index = dict_to_array_and_index(index_dict, variables)
        return snapshots, snapshots_index

    def get_snapshots(LF_DOE_FILE, HF_DOE_FILE, LF_VARIABLES, HF_VARIABLES, HF_TOL_FOR_CONVERGENCE = -8):
        cc2D = ConvergenceChecker(
                residue_function=get_residue_2D
        )

        doe_1D = pd.read_csv(LF_DOE_FILE)

        post_process_1D = DoePostProcessor(
            dataframe = doe_1D,
            snapshot_function = get_1D_snapshot,
        )

        doe_2D = pd.read_csv(HF_DOE_FILE)
        doe_2D = cc2D.check_dataframe(
            dataframe=doe_2D, 
            tol=HF_TOL_FOR_CONVERGENCE,
        )

        post_process_2D = DoePostProcessor(
            dataframe = doe_2D,
            snapshot_function = get_2D_snapshot,
        )

        # filder DOE files to get only converged design points
        post_process_2D.filter( filter = {'converged':True})
        post_process_1D.filter_index(post_process_2D.df['design_point'])


        snapshots_1D = post_process_1D.get_snapshots(
            variables=LF_VARIABLES
        )

        snapshots_2D = post_process_2D.get_snapshots(
            variables = HF_VARIABLES,
            dimensionalization={'Pressure' : 'bc_p0', 'Temperature' : 'bc_T0'}
        )

        return snapshots_1D, snapshots_2D, post_process_1D, post_process_2D

    hyperopt_path = PATH
    Path(hyperopt_path).mkdir(parents=True, exist_ok=True)
        
    # get snapshots
    snapshots_X, snapshots_y, post_process_X, post_process_y = get_snapshots(LF_DOE_FILE, HF_DOE_FILE, LF_VARIABLES, HF_VARIABLES, HF_TOL_FOR_CONVERGENCE)

    post_process_X.df['idx_dict'] = post_process_X.idx_dict.__str__()
    post_process_X.df['snapshots'] = [sX for sX in snapshots_X]

    post_process_y.df['idx_dict'] = post_process_y.idx_dict.__str__()
    post_process_y.df['snapshots'] = [sy for sy in snapshots_y]

    # Split in training, validation and test
    snapshots_X_train_df, snapshots_y_train_df, snapshots_X_test_df, snapshots_y_test_df, VALIDATION_DATA_df = split_dataset(TEST_RATIO, VALIDATION_RATIO, post_process_X.df, post_process_y.df)

    
    snapshots_X_train = np.array(snapshots_X_train_df['snapshots'].tolist())
    snapshots_y_train = np.array(snapshots_y_train_df['snapshots'].tolist())

    np.savez(Path(hyperopt_path) /'training_X.npz', snapshots=snapshots_X_train, doe_file=LF_DOE_FILE, doe_index=snapshots_X_train_df.index, snapshot_index=post_process_X.idx_dict)
    np.savez(Path(hyperopt_path) /'training_y.npz', snapshots=snapshots_y_train, doe_file=HF_DOE_FILE, doe_index=snapshots_y_train_df.index, snapshot_index=post_process_y.idx_dict)

    if snapshots_X_test_df.size > 0:
        snapshots_X_test = np.array(snapshots_X_test_df['snapshots'].tolist())
        snapshots_y_test = np.array(snapshots_y_test_df['snapshots'].tolist())
        np.savez(Path(hyperopt_path) /'test_X.npz', snapshots=snapshots_X_test, doe_file=LF_DOE_FILE, doe_index=snapshots_X_test_df.index, snapshot_index=post_process_X.idx_dict)
        np.savez(Path(hyperopt_path) /'test_y.npz', snapshots=snapshots_y_test, doe_file=HF_DOE_FILE, doe_index=snapshots_y_test_df.index, snapshot_index=post_process_y.idx_dict)
 
    if VALIDATION_DATA_df[0].size > 0 and VALIDATION_DATA_df[1].size > 0:
        VALIDATION_DATA = (np.array(VALIDATION_DATA_df[0]['snapshots'].tolist()), np.array(VALIDATION_DATA_df[1]['snapshots'].tolist()))
        np.savez(Path(hyperopt_path) /'validation_X.npz', snapshots=VALIDATION_DATA[0], doe_file=LF_DOE_FILE, doe_index=VALIDATION_DATA_df[0].index, snapshot_index=post_process_X.idx_dict)
        np.savez(Path(hyperopt_path) /'validation_y.npz', snapshots=VALIDATION_DATA[1], doe_file=HF_DOE_FILE, doe_index=VALIDATION_DATA_df[1].index, snapshot_index=post_process_y.idx_dict)

    

class Indexer:
    def __init__(self, snapshots, idx_dict):
        self.snapshots = snapshots
        self.idx_dict = idx_dict

    def __getitem__(self, idx : list[str]):
        for i in idx:
            if i not in self.idx_dict.keys():
                raise KeyError(f"{i} not in index")
        return np.concatenate([self.snapshots[..., self.idx_dict[i]] for i in idx], axis=-1)
        