import torch
import cv2
import numpy as np

from CRISP.tests.UNet.net import UNet

import sys, os
sys.path.append(os.path.join(os.getcwd(), 'CRISP', 'tests', 'UNet'))

def preprocess(im_):
    im = cv2.resize(im_, (1024,512))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im.astype('float32') / 255.0 - mean) / std
    im = np.rollaxis(im, 2, 0)
    x = torch.from_numpy(im).float()
    x = x.to('cuda', dtype = torch.float)
    x = x.unsqueeze(0)
    return x
    
def predict(model, im):
    model.eval()
    x = torch.from_numpy(im).float()
    x = x.unsqueeze(0)
    with torch.no_grad():
        y = model(x)
    return y

def postprocess(x):
    p = x
    y = torch.sigmoid(p).to('cpu').numpy()
    masks = []
    for channel in y[0]:
        masks.append(channel)
    return masks

def getModel():
    chkp = os.getcwd() + '\\CRISP\\tests\\UNet\\model_e988_gs183768_mIOU92.89.pth'
    model = UNet(9,2)
    model.load_state_dict(torch.load(chkp, map_location=torch.device('cpu'), weights_only=True)['model'])
    model.to(device=torch.device('cuda:0'))
    return model
