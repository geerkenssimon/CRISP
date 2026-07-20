from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torchvision.transforms.functional as f


class RFAnalyzer:
    def __init__(self, *args, **kwargs):
        pass

    def analyze(self, inputs:Union[List[torch.Tensor], torch.Tensor, Tuple[torch.Tensor]]) -> \
        Tuple[List[torch.Tensor], List[torch.Tensor], List[int], List[int], List[int], List[int]]:
        raise NotImplementedError

class NormalRFAnalyzer(RFAnalyzer):
    def __init__(self, model:nn.Module, padding:bool):
        super(NormalRFAnalyzer, self).__init__()
        self.model = model
        self.hooks = []
        self.rf_info = {}  # Final structured result: {input_id: {output_id: info}}
        self._clear_tracking()
        self.i = 0
        self.padding = padding
        
        # Save original methods
        self._orig_cat = torch.cat
        self._orig_add = torch.Tensor.__add__
        self._orig_add_func = torch.add

        self._monkey_patch_concat()
        self._monkey_patch_add()

    def _monkey_patch_concat(self):
        """ 
        Monkey patch torch.cat to keep track of the receptive field of the input tensors.
        The receptive field is stored in a dictionary, where the keys are the ids of the tensors
        and the values are dictionaries with the keys 'rf', 'ef', 'stride', 'inv_stride' and 'padding'.

        The receptive field of the output tensor is the maximum of the receptive fields of the input tensors.
        The effective field of view of the output tensor is the maximum of the effective fields of view of the input tensors.
        The stride of the output tensor is the stride of the first input tensor.
        The inverse stride of the output tensor is the inverse stride of the first input tensor.
        The padding of the output tensor is the maximum of the paddings of the input tensors.
        """
        def cat_hook(inputs, dim=0):
            out = self._orig_cat(inputs, dim=dim)
            infos = [self._tensor_info.get(id(i)) for i in inputs if id(i) in self._tensor_info]
            if infos:
                rf = max(i['rf'] for i in infos)
                ef = max(i['ef'] for i in infos)
                padding = max(i['padding'] for i in infos)
                stride = infos[0]['stride']
                inv_stride = infos[0]['inv_stride']
                self._tensor_info[id(out)] = {
                    'rf': int(rf), 'ef': int(ef),
                    'stride': int(stride), 'inv_stride': int(inv_stride),
                    'padding': int(padding)
                }
            return out
        torch.cat = cat_hook

    def _monkey_patch_add(self):
        """
        Monkey patches torch.add and torch.Tensor.__add__ to track the receptive field 
        when tensors are added. It updates the receptive field information for the 
        output tensor based on the input tensors' receptive field data. This includes 
        the receptive field size ('rf'), effective field of view ('ef'), stride, 
        inverse stride, and padding. The receptive field information is stored in a 
        dictionary where keys are tensor ids and values are dictionaries containing 
        these parameters.
        """
        def add_hook(self_tensor, other):
            out = self._orig_add(self_tensor, other)
            infos = [self._tensor_info.get(id(t)) for t in [self_tensor, other] if id(t) in self._tensor_info]
            if infos:
                rf = max(i['rf'] for i in infos)
                ef = max(i['ef'] for i in infos)
                padding = max(i['padding'] for i in infos)
                stride = infos[0]['stride']
                inv_stride = infos[0]['inv_stride']
                self._tensor_info[id(out)] = {
                    'rf': int(rf), 'ef': int(ef),
                    'stride': int(stride), 'inv_stride': int(inv_stride),
                    'padding': int(padding)
                }
            return out
        torch.add = add_hook

        def add_hook(self_tensor, other):
            out = self._orig_add_func(self_tensor, other)
            infos = [self._tensor_info.get(id(t)) for t in [self_tensor, other] if id(t) in self._tensor_info]
            if infos:
                rf = max(i['rf'] for i in infos)
                ef = max(i['ef'] for i in infos)
                padding = max(i['padding'] for i in infos)
                stride = infos[0]['stride']
                inv_stride = infos[0]['inv_stride']
                self._tensor_info[id(out)] = {
                    'rf': int(rf), 'ef': int(ef),
                    'stride': int(stride), 'inv_stride': int(inv_stride),
                    'padding': int(padding)
                }
            return out
        torch.Tensor.__add__ = add_hook

    def restore(self):
        """
        Restore the original methods of torch.cat, torch.add, and torch.Tensor.__add__
        and remove all the hooks registered by this class. This method is useful when
        you want to stop tracking the receptive field of the input tensors.
        """
        torch.cat = self._orig_cat
        torch.Tensor.__add__ = self._orig_add
        torch.add = self._orig_add_func
        for h in self.hooks:
            h.remove()
        self.i = 0

    def _clear_tracking(self):
        self._current_input_id = None
        self._tensor_info = {}  # id(tensor): rf_info

    def _get_params(self, layer):
        """
        Get the parameters of a convolutional or pooling layer.

        Parameters
        ----------
        layer : nn.Module
            A convolutional or pooling layer.

        Returns
        -------
        tuple
            A tuple of four elements: kernel size, stride, padding, and dilation.
            If the layer is not a convolutional or pooling layer, it returns None.
        """
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.ConvTranspose2d)):
            k = layer.kernel_size
            s = layer.stride
            p = layer.padding
            d = layer.dilation
            if isinstance(k, int): k = (k, k)
            if isinstance(s, int): s = (s, s)
            if isinstance(p, int): p = (p, p)
            if isinstance(d, int): d = (d, d)
            return k, s, p, d
        return [None], [None], [None], [None]

    def _track_module(self, module:nn.Module):
        """
        Registers a forward hook on the given module to track the receptive field of the 
        input tensor and effective field of the output tensors of the module.

        Args:
            module (nn.Module): The module to track the receptive field of.

        Returns:
            None
        """
        def hook(mod:nn.Module, inputs, output):
            if isinstance(inputs, (tuple, list)):
                inputs = inputs[0]

            in_tensor = inputs
            out_tensor = output
            
            # if it is a single layer and not a container, the layers should modify
            # the parameters
            if list(mod.children()) == []:
                prev = self._tensor_info.get(id(in_tensor), {
                    'rf': 1, 'ef': 1, 'stride': 1, 'inv_stride': 1, 'padding': 0
                })
                info = prev.copy()

                # Handle convolution-like layers
                params = self._get_params(mod)
                if isinstance(mod, nn.Upsample):
                    scale = mod.scale_factor
                    info['inv_stride'] = info['inv_stride'] * int(scale)
                    info['ef'] = info['ef'] * int(scale)

                elif isinstance(mod, nn.ConvTranspose2d):
                    k, s, p, d = self._get_params(mod)
                    if self.i == 0 and self.padding:
                        info['padding'] = p[0]
                    info['inv_stride'] = info['inv_stride'] * s[0]
                    info['ef'] = (info['ef']-1) * s[0] + (k[0] - 1) + 1

                else:
                    params = self._get_params(mod)
                    print(self.padding)
                    if self.i == 0 and self.padding:
                        info['padding'] = p[0]
                    if params:
                        k, s, p, d = params
                        info['rf'] += (k[0] - 1) * d[0] * info['stride']
                        info['stride'] = info['stride'] * s[0]

                self._tensor_info[id(out_tensor)] = {
                    k: v for k, v in info.items()
                }
            # if it is a container, the layers within are already computed and therefore
            # the computed parameters of the output should remain
            else:
                self._tensor_info[id(out_tensor)] = self._tensor_info[id(out_tensor)]
        self.hooks.append(module.register_forward_hook(hook))

    def _pair(self, val):
        if isinstance(val, int): return (val, val)
        return val

    def analyze(self, inputs:Union[List[torch.Tensor], torch.Tensor, Tuple[torch.Tensor]]) -> \
        Tuple[List[torch.Tensor], List[torch.Tensor], List[int], List[int], List[int], List[int]]:
        """Analyze the receptive field of the input tensor and the effective field of the output tensors 
        of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to analyze.

        Returns
        -------
        x : torch.Tensor
            The cropped input tensor to the receptive field.
        y : torch.Tensor
            The cropped output tensor to the effective field.
        rf : int
            The receptive field of the input tensor.
        ef : int
            The effective field of the output tensor.
        input_stride : int
            The stride of the input tensor.
        output_stride : int
            The stride of the output tensor.
        """
        self.model.apply(self._track_module)
        input_list = inputs if isinstance(inputs, (list, tuple)) else [inputs]

        xs, ys, rfs, efs, input_strides, output_strides, ps = [], [], [], [], [], [], []

        for x in input_list:
            self._clear_tracking()
            input_id = id(x)
            self._tensor_info[input_id] = {
                'rf': 1, 'ef': 1, 'stride': 1, 'inv_stride': 1, 'padding': 0
            }

            with torch.no_grad():
                y:torch.Tensor = self.model(*input_list)

            rf = self._tensor_info.get(id(y))['rf']
            rfs.append(rf)
            ef = self._tensor_info.get(id(y))['ef']
            efs.append(ef)
            input_stride = self._tensor_info.get(id(y))['stride']
            input_strides.append(input_stride)
            output_stride = self._tensor_info.get(id(y))['inv_stride']
            output_strides.append(output_stride)

            p = self._tensor_info.get(id(y))['padding']
            ps.append(p)

            input_size = x.size()
            output_size = y.size()
            patches_input = (input_size[2] + 2*p - rf) // input_stride + 1,\
                            (input_size[3] + 2*p - rf) // input_stride + 1
            patches_output = (output_size[2] + 2*p - ef) // output_stride + 1,\
                            (output_size[3] + 2*p - ef) // output_stride + 1
            patches = torch.minimum(torch.tensor(patches_input), torch.tensor(patches_output))
            to_consider_input = (patches - 1) * input_stride + rf
            to_consider_output = (patches - 1) * output_stride + ef

            x = f.pad(x, (p, p, p, p), padding_mode='constant', fill=0)

            x = f.center_crop(x, to_consider_input.tolist()).permute(2,3,0,1)
            y = f.center_crop(y, to_consider_output.tolist()).permute(2,3,0,1)
            xs.append(x)
            ys.append(y)
            
        self.restore()
        return xs, ys, rfs, efs, input_strides, output_strides


class SimpleRFAnalyzer(RFAnalyzer):
    def __init__(self, model:nn.Module, padding:bool):
        super(SimpleRFAnalyzer, self).__init__()
        self.model = model
        self.padding = padding

    def _convert_int(self, input):
        if not isinstance(input, int):
            if isinstance(input, (List, Tuple)):
                return int(input[0])
            return int(input)
        return int(input)
        
    def analyze(self, inputs:Union[List[torch.Tensor], torch.Tensor, Tuple[torch.Tensor]]) -> \
        Tuple[List[torch.Tensor], List[torch.Tensor], List[int], List[int], List[int], List[int]]:
        """Analyze the receptive field of the input tensor and the effective field of the output tensors 
        of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to analyze.

        Returns
        -------
        x : torch.Tensor
            The cropped input tensor to the receptive field.
        y : torch.Tensor
            The cropped output tensor to the effective field.
        rf : int
            The receptive field of the input tensor.
        ef : int
            The effective field of the output tensor.
        input_stride : int
            The stride of the input tensor.
        output_stride : int
            The stride of the output tensor.
        """
        layers_x = inputs
        y:torch.Tensor = self.model(*inputs)

        xs, ys, rfs, efs, input_strides, output_strides, ps = [], [], [], [], [], [], []
        padding = None
        stride = None
        
        if not list(self.model.children()) == []:
            for l, module in enumerate(list(self.model.modules())):
                if hasattr(module, 'stride'):
                    stride = self._convert_int(module.stride)
                    break
            for l, module in enumerate(list(self.model.modules())):
                if hasattr(module, 'padding') and self.padding:
                    padding = self._convert_int(module.padding)
                    break
                elif hasattr(module, 'padding') and not self.padding:
                    padding = 0
            if padding is None:
                assert hasattr(self.model, 'padding'), f'Layer of class {self.model.__class__} has no attribute padding'
                padding = self._convert_int(self.model.padding)
            if stride is None:
                assert hasattr(self.model, 'stride'), f'Layer of class {self.model.__class__} has no attribute stride'
                stride = self._convert_int(self.model.stride)

            for j, x in enumerate(layers_x):
                if x.size(2) >= y.size(2):
                    efs.append(1)
                    output_strides.append(1)
                    rf = abs(((y.size(2) - 1) * stride - 2 * padding) - (x.size(2)))
                    rfs.append(rf)
                    input_strides.append(stride)
                    ps.append(padding)
                else:
                    rfs.append(1)
                    input_strides.append(1)
                    output_stride = stride * (max(y.size(2), x.size(2)) // min(y.size(2), x.size(2)))
                    output_strides.append(output_stride)
                    ps.append(padding)
                    ef = abs((x.size(2) - 1) * output_stride - (y.size(2)))
                    efs.append(ef)

        else:
            if isinstance(self.model, nn.Conv2d):
                rfs = [self.model.kernel_size[0]]
                efs = [1]
                input_strides = [self.model.stride[0]]
                output_strides = [1]
                ps = [self._convert_int(self.model.padding)] if self.padding else [0]
            elif isinstance(self.model, nn.ConvTranspose2d):
                rfs = [1]
                efs = [self.model.kernel_size[0]]
                input_strides = [1]
                output_strides = [self.model.stride[0]]
                ps = [self._convert_int(self.model.padding)] if self.padding else [0]

        for i, x in enumerate(layers_x):
            if self.padding:
                x = f.pad(x, ps[i], padding_mode='constant', fill=0)
            else:
                _start = max(ps[i], ps[i] - input_strides[i])
                if _start != 0:
                    x = x[:,_start:-1-_start, _start:-1-_start]
            
            if rfs[i] >= efs[i]:
                _to_consider_x_w = rfs[i] + (y.size(2) - 1) * input_strides[i]
                _to_consider_x_h = rfs[i] + (y.size(3) - 1) * input_strides[i]
                if not padding:    
                    _to_consider_y_w = (x.size(2) - rfs[i]) // input_strides[i] + 1# - torch.max(_to_consider_x - torch.tensor(x.size()[1:]), torch.zeros(2, dtype=torch.int))
                    _to_consider_y_h = (x.size(3) - rfs[i]) // input_strides[i] + 1# - torch.max(_to_consider_x - torch.tensor(x.size()[1:]), torch.zeros(2, dtype=torch.int))
                else:
                    _to_consider_y_w = y.size(2) - max(_to_consider_x_w - x.size(2), 0)
                    _to_consider_y_h = y.size(3) - max(_to_consider_x_h - x.size(3), 0)
            else:
                _to_consider_y_w = efs[i] + (x.size(2) - 1) * output_strides[i]
                _to_consider_y_h = efs[i] + (x.size(3) - 1) * output_strides[i]
                _to_consider_x_w = x.size(2) - max(_to_consider_y_w - y.size(2), 0)
                _to_consider_x_h = x.size(3) - max(_to_consider_y_h - y.size(3), 0)
            to_consider_input = [min(_to_consider_x_w, x.size(2)), min(_to_consider_x_h, x.size(3))]
            to_consider_output = [min(_to_consider_y_w, y.size(2)), min(_to_consider_y_h, y.size(3))]
            
            _x = f.center_crop(x, to_consider_input).permute(2,3,0,1)
            _y = f.center_crop(y, to_consider_output).permute(2,3,0,1)
            
            xs.append(_x)
            ys.append(_y)

        return xs, ys, rfs, efs, input_strides, output_strides