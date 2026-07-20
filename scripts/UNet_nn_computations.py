import sys, os
import torch.nn as nn

import cv2

sys.path.append(os.getcwd())

import CRISP.tests.UNet.model_utils as model_utils
from CRISP.tests.UNet.net import Up, Down
from CRISP.analysis.hidden_layer_pex.computations import _per_layer_SHLQI2

model = model_utils.getModel()
model.eval()
_fn_read_image = cv2.imread

_per_layer_SHLQI2(model=model,
                    _fn_preprocess=model_utils.preprocess,
                    _fn_read_im = _fn_read_image,
                    im_size=(1024,512),
                    stride_x_mul=[3,3,3,3,3,2,2,1,1,1,1,1,1,1,1,1,1,2,2,3,3,5],
                    stride_y_mul=[3,3,3,3,3,2,2,1,1,1,1,1,1,1,1,1,1,2,2,3,3,5],
                    data_location=os.path.join(os.getcwd(), 'CRISP', 'tests', 'UNet', 'images'),
                    save_location=os.path.join(os.getcwd(), 'CRISP', 'tests', 'UNet','QI2_files'),
                    layer_type=nn.Conv2d,
                    max_neighbors=1000,
                    analysis='simple',)

_per_layer_SHLQI2(model=model,
                    _fn_preprocess=model_utils.preprocess,
                    _fn_read_im = _fn_read_image,
                    im_size=(1024,512),
                    stride_x_mul=[1,1,2,3],
                    stride_y_mul=[1,1,2,3],
                    data_location=os.path.join(os.getcwd(), 'CRISP', 'tests', 'UNet', 'images'),
                    save_location=os.path.join(os.getcwd(), 'CRISP', 'tests', 'UNet','QI2_files'),
                    layer_type=Up,
                    max_neighbors=1000,
                    analysis='simple',)

_per_layer_SHLQI2(model=model,
                    _fn_preprocess=model_utils.preprocess,
                    _fn_read_im = _fn_read_image,
                    im_size=(1024,512),
                    stride_x_mul=[2,1,1,1],
                    stride_y_mul=[2,1,1,1],
                    data_location=os.path.join(os.getcwd(), 'CRISP', 'tests', 'UNet', 'images'),
                    save_location=os.path.join(os.getcwd(), 'CRISP', 'tests', 'UNet','QI2_files'),
                    layer_type=Down,
                    max_neighbors=1000,
                    analysis='simple',)

