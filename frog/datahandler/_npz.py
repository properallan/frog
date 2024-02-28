import numpy as np
from ._array import IndexedArray
from ._snapshots import get_snapshot_end_index

class DataHandlerNpz:
    def __init__(self, npz_file):

        self.npz_file = npz_file
        data = np.load(npz_file, allow_pickle=True)
        self.snapshot_index = data['snapshot_index'].item()
        self.snapshot_index_values = list(data['snapshot_index'].item().values())
        self.snapshot_index_keys = list(data['snapshot_index'].item().keys())
        self.doe_index = data['doe_index']
        self.doe_file = data['doe_file'].item()
        self.variables = self.snapshot_index_keys

        self.data = IndexedArray(data['snapshots'],self.snapshot_index, self.doe_index, self.doe_file)

    def __getitem__(self, key): 
        return self.data[key]
        
        #if not isinstance(key, str) and not np.all([isinstance(el,str) for el in key]):
        #    data, index = get_snapshot_end_index(self.npz_file, self.variables)
        #else:
        #    data, index = get_snapshot_end_index(self.npz_file, key)
        #doe_index = self.doe_index
        #doe_file = self.doe_file
        #return IndexedArray(data, index, doe_index, doe_file) 