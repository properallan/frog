import numpy as np
from typing import Union, Iterable

class IndexedArray(np.ndarray):
    def __new__(cls, input_array, index=None, doe_index=None, doe_file=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.index = index
        obj.doe_index = doe_index
        obj.doe_file = doe_file
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.index = getattr(obj, 'index', None)
        self.doe_index = getattr(obj, 'doe_index', None)
        self.doe_file = getattr(obj, 'doe_file', None)

    def __getitem__(self, index):  
        if isinstance(index, str):
            obj = super(IndexedArray, self).__getitem__(Ellipsis)  
            obj = obj[...,self.index[index]]
            obj.index = {index: slice(0,obj.shape[-1],None)}
        elif isinstance(index, Iterable) and np.all([isinstance(el,str) for el in index]):
            obj = super(IndexedArray, self).__getitem__(Ellipsis)  
            obj_ = []
            ind_ = {}
            query_size_1 = 0
            query_size_2 = 0
            for el in index:
                query = obj[...,self.index[el]]
                query_size_2 += query.shape[-1]
                obj_.append(query)
                ind_.update({el: slice(0+query_size_1, query_size_2 ,None)})
                query_size_1 += query.shape[-1]
                
            obj = np.concatenate(obj_, axis=1)
            obj = IndexedArray(obj, ind_, self.doe_index, self.doe_file)
        else:
            obj = super(IndexedArray, self).__getitem__(index)  
            try:
                if not isinstance(index, Iterable):
                    obj = IndexedArray(obj, obj.index, obj.doe_index[index], obj.doe_file)
                else:
                    obj = IndexedArray(obj, obj.index, obj.doe_index[index[0]], obj.doe_file)
            except:
                pass    
        return obj