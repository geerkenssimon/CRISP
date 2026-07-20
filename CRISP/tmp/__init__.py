# built-in modules
import pickle
from typing import Union

# thirs-party modules
import torch

def save(data:Union[dict, object], path:str, sparse:bool=False):
    if sparse:
        if type(data) == dict:
            for key, value in data.items():
                if type(value) == torch.Tensor:
                    data[key] = value.to_sparse_csr()
        if isinstance(data, object):
            for key, value in zip(dir(data), [getattr(data, v) for v in dir(data)]):
                if type(value) == torch.Tensor:
                    if (value == 0).sum() >= (value != 0).sum():
                        setattr(data, key, value.to_sparse_csr())
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    f.close()

def load(file:str, desparse:bool=True):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    f.close()
    if desparse:
        if type(data) == dict:
            for key, value in data.items():
                if type(value) == torch.Tensor:
                    data[key] = value.to_dense().cpu()
        if isinstance(data, object):
            for key, value in zip(dir(data), [getattr(data, v) for v in dir(data)]):
                if type(value) == torch.Tensor:
                    setattr(data, key, value.to_dense().cpu())
    return data

