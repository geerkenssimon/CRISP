# third party modules
import torch
import torch.nn.functional as F

class Euclid:
    def __init__(self, _=None):
        pass

    @staticmethod
    def distance(data:torch.Tensor, secondary_data:torch.Tensor=None) -> torch.Tensor:
        if secondary_data is None:
            return torch.cdist(data, data).double()
        else:
            return torch.cdist(data, secondary_data).float()

class Manhatten:
    def __init__(self, _=None):
        pass

    @staticmethod
    def distance(data:torch.Tensor, secondary_data:torch.Tensor=None) -> torch.Tensor:
        if secondary_data is None:
            return torch.cdist(data, data, p=1).float()
        else:
            return torch.cdist(data, secondary_data, p=1).float()

class Hamming:
    def __init__(self, _=None):
        pass
    
    @staticmethod
    def distance(data:torch.Tensor, secondary_data:torch.Tensor=None) -> torch.Tensor:
        if secondary_data is None:
            return torch.cdist(data, data, p=0).float()
        else:
            return torch.cdist(data, secondary_data, p=0).float()

class Cosine:
    def __init__(self, _=None):
        pass

    @staticmethod
    def distance(data:torch.Tensor, secondary_data:torch.Tensor=None) -> torch.Tensor:
        if secondary_data is None:
            distances = torch.zeros((data.size(0), data.size(0))).to(data.device)
            for i in range(data.size(0)):
                distances[i] = 1 - torch.cosine_similarity(data[i:i+1], data, dim=1).float()
            return distances
        else:
            #distances = torch.zeros((data.size(0), secondary_data.size(0))).to(data.device)
            #for i in range(data.size(0)):
            #    distances[i] = 1 - torch.cosine_similarity(data[i:i+1], secondary_data, dim=1).float()
            #return distances

            distances_ = 1 - (1 + F.cosine_similarity(data[:,None,:], secondary_data[None,:,:], dim=-1).float()) / 2
            return distances_
        
class Dot:
    def __init__(self, _=None):
        pass

    @staticmethod
    def distance(data:torch.Tensor, secondary_data:torch.Tensor=None) -> torch.Tensor:
        if secondary_data is None:
            return torch.mm(data, data.T)
        else:
            return torch.mm(data, secondary_data.T)
