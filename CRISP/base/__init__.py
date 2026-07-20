import torch

if torch.cuda.is_available():
    DEVICE_COUNT = torch.cuda.device_count()
else:
    DEVICE_COUNT = "cpu"

from CRISP.base.SHLQI2 import SHLQI2

from CRISP.base.QI2R import QI2R, DQI2R, VQI2R

__all__ = ['SHLQI2',
           'QI2R',
           'DQI2R',
           'VQI2R'    
]