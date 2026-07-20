# build-in modules
import psutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

# third party modules
import torch

@dataclass
class Points:
    device: int
    start: int
    end: int


def distance_splitting(data:torch.Tensor, devices_available:List) ->  Tuple[Dict[int, Points], List, int]:
    # compute the possible points per device (ppd) based on the available memory  
    cpu_mem = psutil.virtual_memory()
    cpu_mem = [cpu_mem.available / len(devices_available) for device in devices_available]
    intermediate_result_size = torch.tensor([mem *8/32 // data.size(0) for mem in cpu_mem])
    if devices_available != ["cpu"]:
        free_mem = [(torch.cuda.mem_get_info(device)[0] * 0.9) for device in devices_available]
    else:
        free_mem = [psutil.virtual_memory().available * 0.9]
    ppd = torch.tensor([torch.max(torch.tensor(-data.size(1) + (((data.size(1))**2)+(mem*8/32))**(1/2)), 
                        torch.tensor(-data.size(1)/2 - (((data.size(1))**2)+(mem*8/32))**(1/2))) for mem in free_mem])
    ppd = torch.minimum(ppd, intermediate_result_size)
    
    # splitting the dataset based on the possible points per device
    # this splitting can be "evenly" distributed over all devices, if the sum of capacities is higher
    # than the needed capacity. Or it is strictly maximizing the allocated memory per device
    return splitting(ppd, data.size(0), devices_available), free_mem, devices_available


def splitting(possible_ppd:torch.Tensor, remaining_points:int, devices_available:List) -> Dict[int, Points]:
    if possible_ppd.sum() >= remaining_points:
        # if the possible points per device are higher then the datapoints we can evenly devide the dataset onto the devices
        # by considering the respective capacity to the total capacity (in terms of CPU usage we have only 1 device)
        ppd = remaining_points // (possible_ppd.sum()/possible_ppd)
        if devices_available != ["cpu"]:
            point_seperation = {
                i: Points(dev, int(ppd[:i].sum()), int(ppd[:i].sum() + ppd[i])) for i, dev in enumerate(devices_available)
            }
        else:
            point_seperation = {
                i: Points("cpu", int(ppd[:i].sum()), int(ppd[:i].sum() + ppd[i])) for i, _ in enumerate(devices_available)
            }
        remaining_points -= ppd.sum()
        highest_cap = torch.argmax(possible_ppd).item()
        point_seperation[highest_cap].end += int(remaining_points)
    else:
        # if the dataset is bigger than the sum of possible datapoints per device we have to devide the dataset into subsets 
        # regarding the computational capacity of each device. It is possible, that devices have to compute twice or more
        # during the operation. (In terms of CPU usage the CPU has to compute j times)
        i=0
        j=0
        point_seperation = {}
        p_dist = 0
        while remaining_points >= 0:
            if devices_available != ["cpu"] and i >= len(devices_available):
                i = 0
            ppd = possible_ppd[i]
            if devices_available != ["cpu"]:
                point_seperation[j] = Points(devices_available[i], int(p_dist), int(p_dist + ppd if ppd <= remaining_points else p_dist + remaining_points))
            else:
                point_seperation[j] = Points(
                    "cpu", int(p_dist), int(p_dist + ppd if ppd <= remaining_points else p_dist + remaining_points)
                )
            remaining_points -= ppd
            p_dist += ppd
            i+=1
            j+=1
    return point_seperation