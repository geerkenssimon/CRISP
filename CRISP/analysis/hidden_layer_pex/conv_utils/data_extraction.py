# build-in modules
import os

# third-party modules
import torch
import torch.nn as nn

# modules from current project
from CRISP.tmp import save
from CRISP.analysis.hidden_layer_pex.conv_utils.receptivefield import NormalRFAnalyzer, SimpleRFAnalyzer

def _dataset(layer:nn.Module,
            stride_mul:int, 
            save_dest:str, 
            name:str, 
            analysis:str,
            _padding:bool):
    inputs = layer.activations[list(layer.parameters())[0].device]
    # getting 
    # - receptive field (rf)
    # - effective field (ef)
    # - input stride (i_stride)
    # - output stride (o_stride)
    # of the layer regarding the current input
    if analysis == 'normal':
        analyzer = NormalRFAnalyzer(layer, _padding)
        xs, ys, rfs, efs, i_strides, o_strides = analyzer.analyze(inputs)
    elif analysis == 'simple':
        analyzer = SimpleRFAnalyzer(layer, _padding)
        xs, ys, rfs, efs, i_strides, o_strides = analyzer.analyze(inputs)
    for i, (x, y, rf, ef, i_stride, o_stride) in enumerate(zip(xs, ys, rfs, efs, i_strides, o_strides)):
        if x.size(1) == 1:
            continue
        inputs, outputs = _unfold_2D(x, y, rf, ef, i_stride, o_stride, stride_mul)
        save(data= {name + '_in': inputs,
                name + '_out': outputs},
            path = os.path.join(os.getcwd(), save_dest, name) + f'_input_{i}.pkl', sparse=True)
    torch.cuda.empty_cache()

def _unfold_2D(x:torch.Tensor, y:torch.Tensor, 
            rf:int, ef:int, 
            i_stride:int, o_stride:int,
            stride_mul:int):
    import copy 
    _i_stride_ = copy.deepcopy(i_stride)
    _o_stride_ = copy.deepcopy(o_stride)
    _i_stride_ *= stride_mul
    _o_stride_ *= stride_mul

    # unfold input and output to rectangular patches regarding receptive and effective field
    inputs = x.unfold(0, rf, _i_stride_).unfold(1, rf, _i_stride_)
    inputs = inputs.flatten(2).contiguous().view(-1, torch.prod(torch.tensor(inputs.size()[2:])))
    outputs = y.unfold(0, ef, _o_stride_).unfold(1, ef, _o_stride_)
    outputs = outputs.flatten(2).contiguous().view(-1, torch.prod(torch.tensor(outputs.size()[2:])))

    # remove all fully 0 rows regarding the input
    outputs = outputs[~torch.all(inputs == 0, axis=1)].detach().cpu()
    inputs = inputs[~torch.all(inputs == 0, axis=1)].detach().cpu()
    del x, y
    torch.cuda.empty_cache()
    return inputs, outputs



