# build-in modules
from typing import Tuple, Union

# third-party modules
import torch

def distance(data:torch.Tensor, device:str) -> torch.Tensor:
    """Pairwise euclidean distance computation.

    Args:
        data (torch.Tensor): Data to compute the distances on
        device (str): Device ID, where to compute on

    Returns:
        torch.Tensor: Tensor of pairwise euclidean distances
    """
    distances = torch.zeros((data.size(0), data.size(1), data.size(1)), dtype=torch.float32).to(torch.device(device))
    for i in range(data.size(0)):
        distances[i] = torch.cdist(data[i], data[i])
    return distances

def normalize_avg(data:torch.Tensor) -> torch.Tensor:
    """Mean normalization of the data given

    Args:
        data (torch.Tensor): Data to normalize.

    Returns:
        torch.Tensor: Normed data.
    """
    normed = data/((data.sum()+1e-16) / data.size(0)**2)
    return normed

def mlqi_dist(mlqi2:torch.Tensor, sdRE:torch.Tensor, sdRA:torch.Tensor, dist_samples:int) -> Union[torch.Tensor, int]:
    """computational method of the MLQI².
    Additionally added a method called "P-factor method" to randomly stop computation the more neighborhoods
    have been computed.

    Args:
        mlqi2 (torch.Tensor): Tensor representing the empty MLQI².
        sdRE (torch.Tensor): Tensor of inputrelated sorted pairwise euclidean distances.
        sdRA (torch.Tensor): Tensor of outputrelated sorted pairwise euclidean distances.
        inds (torch.Tensor): Tensor of unique indices from BLQI² (to get rid of computational overhead)
        p (float): P-factor for abortion.
        i (int): indice of current datapoint to compute the MLQI² trace
        dist_samples (torch.Tensor): Tensor representing the distances to compute the MLQI² on

    Returns:
        torch.Tensor: Computed MLQI².
    """
    for k_iter, dist in enumerate(dist_samples):
        k = len(torch.where(sdRE[0] <= dist)[0])
        if k_iter >= 1 and k == prev_k:
            mlqi2[k_iter] = mlqi2[k_iter-1]
        else:
            mlqi2[k_iter] = ((normalize_avg(sdRE[:k,:k]) - normalize_avg(sdRA[:k,:k])) ** 2).sum() / k**2
        prev_k = k
    return mlqi2

def mlqi(mlqi2:torch.Tensor, sdRE:torch.Tensor, sdRA:torch.Tensor) -> torch.Tensor:
    """computational method of the MLQI².
    Additionally added a method called "P-factor method" to randomly stop computation the more neighborhoods
    have been computed.

    Args:
        mlqi2 (torch.Tensor): Tensor representing the empty MLQI².
        sdRE (torch.Tensor): Tensor of inputrelated sorted pairwise euclidean distances.
        sdRA (torch.Tensor): Tensor of outputrelated sorted pairwise euclidean distances.
        p (float): P-factor for abortion.

    Returns:
        torch.Tensor: Computed MLQI².
    """
    ci = torch.zeros_like(mlqi2)
    co = torch.zeros_like(mlqi2)
    sdRE_ss = torch.zeros_like(mlqi2)
    sdRA_ss = torch.zeros_like(mlqi2)
    sdREA_s = torch.zeros_like(mlqi2)

    q = (torch.arange(sdRE.size(0), device=mlqi2.device)+1)**2
    
    a = torch.triu((sdRE.T+sdRE))
    b = torch.triu((sdRA.T+sdRA))
    a = torch.sum(a, 0) - torch.diagonal(a)/2
    b = torch.sum(b, 0) - torch.diagonal(b)/2

    ci = torch.cumsum(a, dim=0)
    co = torch.cumsum(b, dim=0)

    a = sdRE**2
    b = sdRA**2
    c = sdRE*sdRA

    a = torch.triu((a.T+a))
    b = torch.triu((b.T+b))
    c = torch.triu((c.T+c))
    
    a = torch.sum(a, 0) - torch.diagonal(a)/2
    b = torch.sum(b, 0) - torch.diagonal(b)/2
    c = torch.sum(c, 0) - torch.diagonal(c)/2

    sdRE_ss = torch.cumsum(a, dim=0)
    sdRA_ss = torch.cumsum(b, dim=0)
    sdREA_s = torch.cumsum(c, dim=0)

    ci = 1/(ci+1e-16)
    co = 1/(co+1e-16)
    
    mlqi2 = q*((ci**2)*sdRE_ss - ci*co*2*sdREA_s + (co**2)*sdRA_ss)
    return mlqi2

def hlqi(HLQI2:torch.Tensor, MLQI2:torch.Tensor, BLQI2:torch.Tensor, minhl:int, maxhl:int, binsizehl:float, path:torch.Tensor, v:int) -> Tuple[torch.Tensor]:
    """Computation of the HLQI².

    Args:
        HLQI2 (torch.Tensor): Empty matrix of HLQI².
        MLQI2 (torch.Tensor): Computed MLQI².
        BLQI2 (torch.Tensor): Computed BLQI².
        minhl (int): Minimum considered Bin.
        maxhl (int): Maximum considered Bin.
        binsizehl (float): Size of the Bins the MLQI² is sorted to.
        path (torch.Tensor): Path of Bins over each neighborhood.
        v (int): Currently computed Bin.

    Returns:
        Tuple[torch.Tensor]: HLQI² and path
    """
    i3 = torch.zeros_like(MLQI2, dtype=torch.int8)
    h = ((MLQI2)-minhl)/binsizehl
    i3[torch.where((h >= v) & (h < v+1))]=1
    HLQI2 = (i3 * BLQI2).sum(axis=0)
    path[i3==1] = v
    return HLQI2, path