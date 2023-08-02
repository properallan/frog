import h5py
import numpy as np
from collections.abc import Iterable
from typing import Any, Union

class HDF5HandlerTransposedOldShape:
    """
    Class to handle HDF5 files. For old snapshot shape (n_variables, n_points).

    Parameters
    ----------
    datapath : str
        Path to the HDF5 file.
    datasets : list, optional
        List of datasets to load. If None, all datasets are loaded.
    
    Attributes
    ----------
    datasets : list
        List of datasets in the HDF5 file.
    data : np.ndarray
        Array containing the data of all datasets.
    start_indexes : dict
        Dictionary containing the start indexes of each dataset in the data array.
    end_indexes : dict
        Dictionary containing the end indexes of each dataset in the data array.
    lenghts : list
        List containing the lenghts of each dataset.

    Methods
    -------
    __init__(self, datapath : str, datasets : list = None) -> np.ndarray
        Initialize the HDF5Handler.
    _load_data(self, datasets: list = None)
        Load the data from the HDF5 file.
    _slice(self, index : str) -> slice
        Return a slice of the data array.
    __getitem__(self, index : Union[str, list, tuple]) -> np.ndarray
        Return a slice of the data array.

    Examples
    --------
    
    """

    def __init__(self, datapath : str, datasets : list = None) -> None:
        """Initialize the HDF5Handler.
        
        Args:
            datapath (str): Path to the HDF5 file.
            datasets (list, optional): List of datasets to load. If None, all datasets are loaded.
        """

        self.datapath : str = datapath
        self.data : np.dnarray = None
        self._load_data(datasets)

    def _load_data(self, datasets: list = None) -> np.ndarray:
        """Load the data from the HDF5 file.
        
        Args:
            datasets (list, optional): List of datasets to load. If None, all datasets are loaded.
            
        Returns:
            np.ndarray: Array containing the data of all datasets.
        """

        with h5py.File(self.datapath, 'r') as f:
            if datasets is None:
                datasets = list(f.keys())
                # select only numerical datasets
                datasets = [var for var in datasets if type(
                    f[var]) == h5py.Dataset]
            self.datasets = datasets

            # select data if it is a dataset
            self.data = [f[var][:] for var in datasets]

        # reshape data if it is 1D
        self.data = [data[None, :] if data.ndim ==
                     1 else data for data in self.data]

        self.start_indexes = {}
        self.end_indexes = {}

        self.lenghts = [data.shape[0] for data in self.data]

        last_index = 0
        for i, var in enumerate(self.datasets):
            self.start_indexes[var] = 0 + last_index
            self.end_indexes[var] = self.lenghts[i] + last_index
            last_index += self.lenghts[i]

        # concatenate data in a single snapshots
        self.data = np.concatenate(self.data, axis=0)
        self.data_for_handling = self.data

        return self.data

    def _slice(self, index : str) -> slice:
        """Get a slice index of the corresponding dataset name.

        Args:
            index (str): Name of the dataset.

        Returns:
            slice: Slice index.
        """

        return slice(self.start_indexes[index], self.end_indexes[index], None)

    def __getitem__(self, index : Union[str, list, tuple], data=None) -> np.ndarray:
        """Get a slice of the data array.
        
        Args:
            index (Union[str, list, tuple]): Slice index.
            
        Returns:    
            np.ndarray: Slice of the data array.        
        """

        if isinstance(index, list):
            index = tuple([index, slice(None, None, None)])
        if isinstance(index, str):
            return self.data[self.start_indexes[index]:self.end_indexes[index]]
        if isinstance(index, Iterable):
            index = list(index)
            for i, ind in enumerate(index):
                if type(ind) == str:
                    index[i] = slice(self.start_indexes[ind],
                                     self.end_indexes[ind], None)
                elif type(ind) == list:
                    return np.concatenate([self.__getitem__(tuple([ind_i, *index[1:]])) for ind_i in ind])
            index = tuple(index)
        query = self.data_for_handling[index]
        self.data_for_handling = self.data

        return query
    
    def __call__(self, data) -> None:
        self.data_for_handling = data
    

class HDF5Handler:
    """
    Class to handle HDF5 files.

    Parameters
    ----------
    datapath : str
        Path to the HDF5 file.
    datasets : list, optional
        List of datasets to load. If None, all datasets are loaded.
    
    Attributes
    ----------
    datasets : list
        List of datasets in the HDF5 file.
    data : np.ndarray
        Array containing the data of all datasets.
    start_indexes : dict
        Dictionary containing the start indexes of each dataset in the data array.
    end_indexes : dict
        Dictionary containing the end indexes of each dataset in the data array.
    lenghts : list
        List containing the lenghts of each dataset.

    Methods
    -------
    __init__(self, datapath : str, datasets : list = None) -> np.ndarray
        Initialize the HDF5Handler.
    _load_data(self, datasets: list = None)
        Load the data from the HDF5 file.
    _slice(self, index : str) -> slice
        Return a slice of the data array.
    __getitem__(self, index : Union[str, list, tuple]) -> np.ndarray
        Return a slice of the data array.

    Examples
    --------
    
    """

    def __init__(self, datapath : str, datasets : list = None) -> None:
        """Initialize the HDF5Handler.
        
        Args:
            datapath (str): Path to the HDF5 file.
            datasets (list, optional): List of datasets to load. If None, all datasets are loaded.
        """

        self.datapath : str = datapath
        self.data : np.dnarray = None
        self._load_data(datasets)

    def _load_data(self, datasets: list = None) -> np.ndarray:
        """Load the data from the HDF5 file.
        
        Args:
            datasets (list, optional): List of datasets to load. If None, all datasets are loaded.
            
        Returns:
            np.ndarray: Array containing the data of all datasets.
        """

        with h5py.File(self.datapath, 'r') as f:
            if datasets is None:
                datasets = list(f.keys())
                # select only numerical datasets
                datasets = [var for var in datasets if type(
                    f[var]) == h5py.Dataset]
            self.datasets = datasets

            # select data if it is a dataset
            self.data = [f[var][:] for var in datasets]

        # reshape data if it is 1D
        self.data = [data[:,None] if data.ndim ==
                     1 else data for data in self.data]

        self.start_indexes = {}
        self.end_indexes = {}

        self.lenghts = [data.shape[-1] for data in self.data]

        last_index = 0
        for i, var in enumerate(self.datasets):
            self.start_indexes[var] = 0 + last_index
            self.end_indexes[var] = self.lenghts[i] + last_index
            last_index += self.lenghts[i]

        # concatenate data in a single snapshots
        self.data = np.concatenate(self.data, axis=-1)
        self.data_for_handling = self.data

        return self.data

    def _slice(self, index : str) -> slice:
        """Get a slice index of the corresponding dataset name.

        Args:
            index (str): Name of the dataset.

        Returns:
            slice: Slice index.
        """

        return slice(self.start_indexes[index], self.end_indexes[index], None)
    
    def get_index(self, index : Union[str, list, tuple]) -> np.ndarray:
        """Get a index of the data array.
        
        Args:
            index (Union[str, list, tuple]): Slice index.
            
        Returns:    
            np.ndarray: Slice of the data array.        
        """

        if isinstance(index, list):
            index = tuple([slice(None, None, None), index])
        if isinstance(index, str):
            index = tuple( [Ellipsis , slice(self.start_indexes[index], self.end_indexes[index], None) ])  
        if isinstance(index, Iterable):
            index = list(index)
            for i, ind in enumerate(index):
                if type(ind) == str:
                    index[i] = slice(self.start_indexes[ind],
                                     self.end_indexes[ind], None)
                elif type(ind) == list:
                    return np.concatenate([self.__getitem__(tuple([*index[0:-1], ind_i])) for ind_i in ind], axis=-1)
            index = tuple(index)
        return index

    def __getitem__(self, index : Union[str, list, tuple]) -> np.ndarray:
        """Get a slice of the data array.
        
        Args:
            index (Union[str, list, tuple]): Slice index.
            
        Returns:    
            np.ndarray: Slice of the data array.        
        """
        
        index = self.get_index(index)
        if isinstance(index, np.ndarray):
            return index
        
        query = self.data_for_handling[index]
        self.data_for_handling = self.data  
        return query
    
    def __call__(self, data) -> None:
        self.data_for_handling = data
        return self