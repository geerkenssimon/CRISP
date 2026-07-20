# built-in modules
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

# third-party modules
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.hooks import RemovableHandle
from torch import Tensor

from captum.attr import LRP
from captum.attr._utils.custom_modules import Addition_Module
from captum._utils.common import (
    _format_inputs,
    _format_output,
    _is_tuple,
    _register_backward_hook
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.log import log_usage

# modules from current project
from CRISP.analysis._utils import _run_forward, compute_gradients
from CRISP.analysis._utils.lrp_rules import (
    PropagationRule,
    EpsilonRule, 
    GammaRule, 
    WW_Rule, 
    zBRule, 
    Alpha_Beta_Rule, 
    Alpha1_Beta0_Rule,
    IdentityRule, 
    UpsampleRule,
    ContainerLayerRule,
    NoneRule,
    _print_hook
)

#from captum.attr._utils.lrp_rules import EpsilonRule, PropagationRule

from CRISP.tmp import save


class _LRP(LRP):
    """Extension of the LRP class from captum 'https://github.com/pytorch/captum'.

    This class inherits from the 'LRP' class from captum and extends its functionality in order
    to do a comprehensive SHLQI² based analysis of a network. For this type of analysis, this 
    class needs to be able to create datasets from every specified layer in the network and needs
    to take every layer or layer group as starting point for backward LRP attribution.

    Functionalities added:
        - Naming of layers throughout the network (`_give_names()`)
        - intermediate output size generation (`_intermediate_layer_output_shape()`)
        - setting intermediate hook for targeted backward attribution (`_set_intermediate_hook()`)

    Functionalities modified:
        - wrapper of forward function now incorporates the output of the intermediate hook
            (`_forward_fn_wrapper`)
        - initialisation of added and modified LRP-rules due to recommended LRP_CMP from 
            paper https://arxiv.org/pdf/1910.09840.pdf (`_check_and_attach_rules`)

    Parameters
    ----------
    LRP : class
        _description_
    """
    def __init__(self, model: nn.Module, pixel_values: Tuple[float] = None) -> None:
        super().__init__(model)
        self.verbose = False
        self.pixel_values = pixel_values
        self.intermediate_hook = False
        self.layer_names = []
        self.names_buildup = []
        self.iterator = {key : 0 for key in list(dir(nn.modules)) if key[0].isupper() } #and not key in ['ModuleDict', 'ModuleList', 'Sequential', 'Module', 'OrderedDict']}
        self._give_names(self.model)        # added (see below for function)
        self.iterator = {key : 0 for key in list(dir(nn.modules)) if key[0].isupper() }
        self.names_buildup = []
        self.layers:List[nn.Module] = []
        self._get_layers(self.model)
        self._original_state_dict = self.model.state_dict()
        self.gradient_func = compute_gradients

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
        verbose: bool = False,
        layer: nn.Module = None,
        _restore_model: bool = True
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        r"""
        Args:
            inputs (tensor or tuple of tensors):  Input for which relevance is
                        propagated. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            target (int, tuple, tensor or list, optional):  Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                    - a single integer or a tensor containing a single
                        integer, which is applied to all input examples

                    - a list of integers or a 1D tensor, with length matching
                        the number of examples in inputs (dim 0). Each integer
                        is applied as the target for the corresponding example.

                    For outputs with > 2 dimensions, targets can be either:

                    - A single tuple, which contains #output_dims - 1
                        elements. This target index is applied to all examples.

                    - A list of tuples with length equal to the number of
                        examples in inputs (dim 0), and each tuple containing
                        #output_dims - 1 elements. Each tuple is applied as the
                        target for the corresponding example.

                    Default: None
            additional_forward_args (tuple, optional): If the forward function
                    requires additional arguments other than the inputs for
                    which attributions should not be computed, this argument
                    can be provided. It must be either a single additional
                    argument of a Tensor or arbitrary (non-tuple) type or a tuple
                    containing multiple additional arguments including tensors
                    or any arbitrary python types. These arguments are provided to
                    forward_func in order, following the arguments in inputs.
                    Note that attributions are not computed with respect
                    to these arguments.
                    Default: None

            return_convergence_delta (bool, optional): Indicates whether to return
                    convergence delta or not. If `return_convergence_delta`
                    is set to True convergence delta will be returned in
                    a tuple following attributions.
                    Default: False

            verbose (bool, optional): Indicates whether information on application
                    of rules is printed during propagation.

            layer (str, optional): Indicates the intermediate layer where to start
                    the LRP from. If no string is given, the whole network will be 
                    used for the LRP

            

        Returns:
            *tensor* or tuple of *tensors* of **attributions**
            or 2-element tuple of **attributions**, **delta**::
            - **attributions** (*tensor* or tuple of *tensors*):
                        The propagated relevance values with respect to each
                        input feature. The values are normalized by the output score
                        value (sum(relevance)=1). To obtain values comparable to other
                        methods or implementations these values need to be multiplied
                        by the output score. Attributions will always
                        be the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned. The sum of attributions
                        is one and not corresponding to the prediction score as in other
                        implementations.
            - **delta** (*tensor*, returned if return_convergence_delta=True):
                        Delta is calculated per example, meaning that the number of
                        elements in returned delta tensor is equal to the number of
                        of examples in the inputs.
        Examples:

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities. It has one
                >>> # Conv2D and a ReLU layer.
                >>> net = ImageClassifier()
                >>> lrp = LRP(net)
                >>> input = torch.randn(3, 3, 32, 32)
                >>> # Attribution size matches input size: 3x3x32x32
                >>> attribution = lrp.attribute(input, target=5)

        """        
        self.verbose = verbose
        if layer is not None:
            self.endlayer = layer
        else:
            self.endlayer = list(self.model.modules())[-1]
        
        self._check_and_attach_rules()
        self.backward_handles: List[RemovableHandle] = []
        self.forward_handles: List[RemovableHandle] = []

        is_inputs_tuple = _is_tuple(inputs)
        inputs = _format_inputs(inputs)
        gradient_mask = apply_gradient_requirements(inputs)
        #self._set_intermediate_hook(grad=True)
        try:
            with torch.autograd.set_detect_anomaly(False):
                # 1. Forward pass: Change weights of layers according to selected rules.
                output = self._compute_output_and_change_weights(
                    inputs, target, additional_forward_args
                )
                #if target is None or type(target) == torch.Tensor and target.numel() != 1:
                #    del output
                self._set_intermediate_hook(grad=True)
                # 2. Forward pass + backward pass: Register hooks to configure relevance
                # propagation and execute back-propagation.
                self._register_forward_hooks()
                normalized_relevances = self.gradient_func(
                    self._forward_fn_wrapper, inputs, target, additional_forward_args
                )
                if target is not None and (type(target) == int or target.numel() == 1):
                    relevances = tuple(
                        normalized_relevance
                        * output.reshape((-1,) + (1,) * (normalized_relevance.dim() - 1))
                        for normalized_relevance in normalized_relevances
                    )
                else:
                    relevances = tuple(normalized_relevance for normalized_relevance in normalized_relevances)
        finally:
            if _restore_model:
                self._restore_model()

        undo_gradient_requirements(inputs, gradient_mask)

        if return_convergence_delta:
            return (
                _format_output(is_inputs_tuple, relevances),
                self.compute_convergence_delta(relevances, output),
            )
        else:
            return _format_output(is_inputs_tuple, relevances)  # type: ignore
        


    @log_usage()
    def forward_with_hooks(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        verbose: bool = False,
        layer: nn.Module = None,
        _restore_model: bool = True
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        self.verbose = verbose
        if layer is not None:
            self.endlayer = layer
        else:
            self.endlayer = list(self.model.modules())[-1]
        self._check_and_attach_rules()
        self.backward_handles: List[RemovableHandle] = []
        self.forward_handles: List[RemovableHandle] = []

        inputs = _format_inputs(inputs)
        gradient_mask = apply_gradient_requirements(inputs)
        self._set_intermediate_hook(grad=True)
        try:
            with torch.autograd.set_detect_anomaly(False):
                # 1. Forward pass: Change weights of layers according to selected rules.
                self._compute_output_and_change_weights(
                    inputs, target, additional_forward_args
                )
        finally:
            if _restore_model:
                self._restore_model()

        undo_gradient_requirements(inputs, gradient_mask)

        
    def _get_layers(self, model: nn.Module) -> None:
        for layer in list(self.model.modules())[1:]:
            self.layers.append(layer)

    def _compute_output_and_change_weights(
        self,
        inputs: Tuple[Tensor, ...],
        target: TargetType,
        additional_forward_args: Any,
    ) -> Tensor:
        try:
            self._register_weight_hooks()
            output = _run_forward(self._forward_fn_wrapper, inputs, target, additional_forward_args)
        finally:
            self._remove_forward_hooks()
        # Register pre_hooks that pass the initial activations from before weight
        # adjustments as inputs to the layers with adjusted weights. This procedure
        # is important for graph generation in the 2nd forward pass.
        self._register_pre_hooks()
        return output

    def _set_intermediate_hook(self, grad=False) -> None:
        # changed to usage for intermediate layer lrp
        # Layers following the relevant and chosen layer to start the lrp from 
        # are set to nn.Identity() so that they dont harm the forward_fn from the model
        # but also allow the usage from intermediate layer
        def hook_func(m, inp, op:torch.Tensor):
            if not grad:
                with torch.no_grad():
                    m.op_hook = op.detach().clone()
            else:
                m.op_hook = op

        handle = self.endlayer.register_forward_hook(hook_func)
        
        self.endlayer.handles = []
        #self.endlayer.handles.append(handle)
        self.forward_handles.append(handle)
        self.intermediate_hook = True

    def _give_names(self, model: nn.Module) -> None:
        """Function to assign unique names to layers in a network for identification.
        
        Args:
            model (nn.Module): The neural network model.
        """
        for name, layer in model.named_children():
            layer_key = '-'.join([*self.names_buildup, layer._get_name()])  # Unique key based on hierarchy

            if layer_key not in self.iterator:  # Only initialize if the full path isn't seen before
                self.iterator[layer_key] = 0

            layer_name = f"{layer._get_name()}{self.iterator[layer_key]}"  # Unique layer name
            self.layer_names.append('-'.join([*self.names_buildup, layer_name]))
            self.iterator[layer_key] += 1
            layer.name = self.layer_names[-1]

            self.names_buildup.append(name)

            if len(list(layer.named_children())) > 0:  # Ensure it has children before recursion
                self._give_names(layer)

            if self.names_buildup:  # Maintain correct hierarchy after recursion
                self.names_buildup.pop(-1)

    def _search_layer(self, model:nn.Module, _name: str) -> None:
        """Function to assign unique names to layers in a network for identification.
        
        Args:
            model (nn.Module): The neural network model.
        """
        for name, layer in model.named_children():
            layer_key = '-'.join([*self.names_buildup, layer._get_name()])  # Unique key based on hierarchy

            if layer_key not in self.iterator:  # Only initialize if the full path isn't seen before
                self.iterator[layer_key] = 0

            layer_name = f"{layer._get_name()}{self.iterator[layer_key]}"  # Unique layer name
            if '-'.join([*self.names_buildup, layer_name]) == _name:
                return layer
            self.iterator[layer_key] += 1

            self.names_buildup.append(name)

            if len(list(layer.named_children())) > 0:  # Ensure it has children before recursion
                l = self._search_layer(layer, _name)
                if l is not None:
                    return l

            if self.names_buildup:  # Maintain correct hierarchy after recursion
                self.names_buildup.pop(-1)

    def _check_and_attach_rules(self) -> None:
        first_conv = 0
        for i,layer in enumerate(self.layers):
            if hasattr(layer, "rule") and layer.rule is not None:
                layer.activations = {}  # type: ignore
                layer.outputs = {}
                layer.rule.relevance_input = defaultdict(list)  # type: ignore
                layer.rule.relevance_output = {}  # type: ignore
                pass
            elif type(layer) in SUPPORTED_LAYERS_WITH_RULES.keys():
                if type(layer) == nn.Conv2d and first_conv < 3 and self.pixel_values is None:           # apply w²-rule for non pixel dependant 1st convolutional layer
                    layer.rule = WW_Rule()  # type: ignore
                    first_conv += 1
                elif type(layer) == nn.Conv2d and first_conv < 3 and self.pixel_values is not None:       # apply zB-rule for pixel dependant 1st convolutional layer
                    layer.rule = zBRule(self.pixel_values)
                    first_conv += 1
                #elif type(layer) == nn.Conv2d and i/len(self.layers) <= .4:
                #    layer.rule = GammaRule(gamma=0.25)
                #elif type(layer) == nn.Conv2d and i/len(self.layers) <= .8:
                #    layer.rule = EpsilonRule(epsilon=0.25)
                elif SUPPORTED_LAYERS_WITH_RULES[type(layer)] == Alpha_Beta_Rule:
                    layer.rule = SUPPORTED_LAYERS_WITH_RULES[type(layer)](alpha=2)
                else:
                    layer.rule = SUPPORTED_LAYERS_WITH_RULES[type(layer)]()  # type: ignore
                    
                layer.activations = {}  # type: ignore
                layer.outputs = {}
                layer.rule.relevance_input = defaultdict(list)  # type: ignore
                layer.rule.relevance_output = {}  # type: ignore
            elif type(layer) in SUPPORTED_NON_LINEAR_LAYERS:
                layer.rule = None  # type: ignore
            elif len(list(layer.children())) != 0:
                layer.rule = ContainerLayerRule()
                layer.activations = {}  # type: ignore
                layer.outputs = {}
                layer.rule.relevance_input = defaultdict(list)  # type: ignore
                layer.rule.relevance_output = {}  # type: ignore
            else:
                raise TypeError(
                    (
                        f"Module of type {type(layer)} has no rule defined and no"
                        "default rule exists for this module type. Please, set a rule"
                        "explicitly for this module and assure that it is appropriate"
                        "for this type of layer."
                    )
                )
        self.endlayer.input_grad = defaultdict(list)
        self.endlayer.output_grad = {}

    def _register_forward_hooks(self) -> None:
        for layer in self.layers:
            if len(list(layer.children())) != 0 and not self.endlayer.rule == ContainerLayerRule:
                if self.verbose:
                    forward_handle = layer.register_forward_hook(
                        _print_hook  # type: ignore
                    )
                    self.forward_handles.append(forward_handle)
                continue
            if type(layer) in SUPPORTED_NON_LINEAR_LAYERS:
                if self.verbose:
                    forward_handle = layer.register_forward_hook(
                        _print_hook  # type: ignore
                    )
                    self.forward_handles.append(forward_handle)
                backward_handles = _register_backward_hook(
                    layer, PropagationRule.backward_hook_activation, self
                )
                self.backward_handles.extend(backward_handles)
            else:
                forward_handle = layer.register_forward_hook(
                    layer.rule.forward_hook  # type: ignore
                )
                self.forward_handles.append(forward_handle)
                if self.verbose:
                    print(f"Applied {layer.rule} on layer {layer}")
            if self.verbose:
                forward_handle = layer.register_forward_hook(
                    _print_hook  # type: ignore
                )
                self.forward_handles.append(forward_handle)
            
    def _intermediate_layer_output_shape(self, layer:nn.Module, x:torch.Tensor):
        """compute the output to the specified layer and return its output.

        Args:
            layer (str): Name of the specified intermediate layer
            x (torch.Tensor): Input to the network

        Returns:
            Tuple: Tuple of the output shape
        """
        self.endlayer = layer
        self.backward_handles: List[RemovableHandle] = []
        self.forward_handles: List[RemovableHandle] = []
        self._check_and_attach_rules()
        self._set_intermediate_hook()

        y = self._forward_fn_wrapper(x)

        self._restore_model()

        return y.shape
    
    def _forward_fn_wrapper(self, *inputs: Tensor) -> Tensor:
        """
        Wraps a forward function with addition of zero as a workaround to
        https://github.com/pytorch/pytorch/issues/35802 discussed in
        https://github.com/pytorch/captum/issues/143#issuecomment-611750044

        #TODO: Remove when bugs are fixed
        """
        adjusted_inputs = tuple(
            input + 0 if input is not None else input for input in inputs
        )
        if not self.intermediate_hook:
            return self.model(*adjusted_inputs)
        else:
            self.model(*adjusted_inputs)
            return self.endlayer.op_hook
            
    def generate_dataset(self, 
                         input:torch.Tensor,
                         layer_type:str,
                         save_dest:str,
                         stride_x_mul:Union[int, List] = 1,
                         stride_y_mul:Union[int, List] = 1,
                         _dataset_generation_fn:Union[Callable, None] = None
                         ):
        
        layers_to_compute = [module for module in self.model.modules() if module._get_name() == layer_type]

        assert type(stride_x_mul) == type(stride_y_mul), f'Types of stride multiplicates not equal. \
            Type of stride_x_mul: {type(stride_x_mul)}, type of stride_y_mul {type(stride_y_mul)}.'
        assert type(stride_x_mul) == int or type(stride_x_mul) == list, f'Types of stride multiplicators not \
            valid. Types ({type(stride_x_mul)},{type(stride_y_mul)}) must be either `int` or `list`.'
        if type(stride_x_mul) == list:
            assert len(stride_x_mul) == len(stride_y_mul), f'Length of stride multiplicators does not match. \
                {len(stride_x_mul)}, {len(stride_y_mul)}'
            assert len(stride_x_mul) == len(layers_to_compute)

        for i, layer in enumerate(layers_to_compute):
            self.attribute(input, layer=layer, test=True)

            layer_name = self.layer_names[self.layers.index(layer)]

            for l, module in enumerate(list(layer.modules())):
                if hasattr(module, 'stride'):
                    stride = torch.tensor(module.stride)
                    padding = torch.tensor(module.padding)
                    break
            r = torch.ones_like(torch.tensor(stride))
            for l, module in enumerate(list(layer.modules())):
                s = torch.ones_like(torch.tensor(stride))
                for _module in list(layer.modules())[:l]:
                    if hasattr(_module, 'kernel_size'):
                        s *= torch.tensor(_module.stride)
                if hasattr(module, 'kernel_size'):
                    r = r + (torch.tensor(module.kernel_size) - 1) * s

            if type(stride_x_mul) == list:
                _stride_mul = torch.tensor((stride_x_mul[i],
                                            stride_y_mul[i]))
            else:
                _stride_mul = torch.tensor((stride_x_mul,
                                            stride_y_mul))

            if _dataset_generation_fn is not None:
                _dataset_generation_fn(layer, stride, padding, r, _stride_mul, save_dest, layer_name)
            elif stride.numel() == 1:
                self._dataset_from_1D(layer, stride, padding, r, _stride_mul, save_dest, layer_name)
            else:
                self._dataset_from_2D(layer, stride, padding, r, _stride_mul, save_dest, layer_name)

    def _dataset_from_1D(self, layer:nn.Module, stride:torch.Tensor, padding:torch.Tensor, receptive_field:torch.Tensor, stride_mul:int, save_dest:str, name:str):
        pass

    def _dataset_from_2D(self, layer:nn.Module, stride:torch.Tensor, padding:torch.Tensor, receptive_field:torch.Tensor, stride_mul:int, save_dest:str, name:str):
        device = next(layer.parameters()).device
        x:torch.Tensor = layer.activations[device][0][0]
        y:torch.Tensor = layer.outputs[device][0]

        stride *= stride_mul

        padding_x, padding_y = padding
        x = f.pad(x, (padding_x, padding_x, padding_y, padding_y), 'constant', 0)

        _to_consider = receptive_field + (torch.tensor(y.size()[1:]) - 1) * stride
        x = x[:,:_to_consider[0],:_to_consider[1]].permute(1,2,0)
        y = y.permute(1,2,0)

        inputs = x.unfold(0,receptive_field[0], stride[0]).unfold(1,receptive_field[1], stride[1])
        inputs = inputs.flatten(2).view(torch.prod(torch.tensor(inputs.size()[:2])), -1)

        outputs = y.unfold(0,1,stride_mul[0]).unfold(1,1,stride_mul[1])
        outputs = outputs.flatten(2).view(torch.prod(torch.tensor(outputs.size()[:2])), -1)

        outputs = outputs[~torch.all(inputs == 0, axis=1)]
        inputs = inputs[~torch.all(inputs == 0, axis=1)]
        save(data = {name + '_in': inputs,
                    name + '_out': outputs},
            path = Path(os.getcwd()) / save_dest / (name + '.pkl')
            )


SUPPORTED_LAYERS_WITH_RULES = {
    nn.MaxPool1d: EpsilonRule,
    nn.MaxPool2d: EpsilonRule,
    nn.MaxPool3d: EpsilonRule,
    nn.Conv2d: Alpha_Beta_Rule,
    #nn.Conv2d: FlatRule,
    #nn.Conv2d: GammaRule,
    #nn.Conv2d: EpsilonRule,
    nn.AvgPool2d: EpsilonRule,
    nn.AdaptiveAvgPool2d: EpsilonRule,
    #nn.Linear: FlatRule,
    nn.Linear: EpsilonRule,
    nn.BatchNorm2d: IdentityRule,
    #nn.BatchNorm2d: EpsilonRule,
    nn.Upsample: EpsilonRule,
    Addition_Module: EpsilonRule,
    #nn.Softmax: EpsilonRule,
    #nn.Sequential: IdentityRule
}

SUPPORTED_NON_LINEAR_LAYERS = [nn.ReLU, nn.Dropout, nn.Tanh, nn.Identity, nn.Flatten, nn.Dropout2d, nn.Sigmoid, nn.SiLU, nn.Softmax]

NON_RULE_LAYERS = [
    nn.Sequential,
    nn.ModuleDict,
    nn.ModuleList,
]
