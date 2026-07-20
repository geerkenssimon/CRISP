import math
import torch
import psutil

from CRISP.base._utils.metrics.image import SSIM

from CRISP.base._utils.metrics.vector import Euclid
from CRISP.base._utils.metrics.vector import Hamming
from CRISP.base._utils.metrics.vector import Cosine
from CRISP.base._utils.metrics.vector import Dot
from CRISP.base._utils.metrics.vector import Manhatten

class Metric:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def distance(data:torch.Tensor, secondary_data:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

metrics = {
    'SSIM': SSIM,
    'Euclid': Euclid,
    'Hamming': Hamming,
    'Cosine': Cosine,
    'Dot': Dot,
    'Manhatten': Manhatten,
}

VECTOR_METRICS = {
    "Euclid": Euclid,
    "Hamming": Hamming,
    "Cosine": Cosine,
    "Dot": Dot,
    "Manhatten": Manhatten,
}

IMAGE_METRICS = {"SSIM": SSIM}

def _thread_per_device(metric:Metric, data:torch.Tensor, start:int, end:int, device:int):
    # calculating the allocated and free memory on the respective device by the baseline chunk
    # given from the previous ppd computation. This might be misaligned with full memory 
    # usage if the sum of capacities is higher than the capacity needed for the whole data
    memory_allocated = (end-start)*data.size(1) * 32/8
    if device != "cpu":
        free_mem = torch.cuda.mem_get_info(device)[0] * 0.9 - memory_allocated
    else:
        free_mem = psutil.virtual_memory().available * 0.9 - memory_allocated

    # based on the free memory, the secondary chunk can be bigger than the baseline chunk.
    # Exp.: 6 Devices, evenly distirbuted data splitting. --> looping over the given chunks
    # per device may take longer than comparing the given baseline chunk with more data
    # based on the available device memory
    sec_chunk_size = int(free_mem // ((end-start + data.size(1)) * 32/8))
    sec_start = 0
    distances = torch.zeros((end-start, data.size(0)), dtype=torch.float32)

    for i in range(math.ceil(data.size(0)/sec_chunk_size)):
        distances[:,i*sec_start:i*sec_start+sec_chunk_size] = metric.distance(data[start:end].to(device), 
                                                data[i*sec_start:i*sec_start+sec_chunk_size].to(device)).detach().cpu()
        sec_start += sec_chunk_size 
    #torch.cuda.empty_cache()
    return distances, start, end, device
