# built-in modules
from inspect import signature
from typing import Any, Callable, Tuple, Union, cast

#third-party modules
import torch
from torch import Tensor

from captum._utils.common import _verify_select_column, _format_inputs, _format_additional_forward_args
from captum._utils.typing import (
    TargetType,
)

def _run_forward(
    forward_func: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target: TargetType = None,
    additional_forward_args: Any = None,
) -> Tensor:
    forward_func_args = signature(forward_func).parameters
    if len(forward_func_args) == 0:
        output = forward_func()
        return output if target is None else _select_targets(output, target)

    # make everything a tuple so that it is easy to unpack without
    # using if-statements
    inputs = _format_inputs(inputs)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    output = forward_func(
        *(*inputs, *additional_forward_args)
        if additional_forward_args is not None
        else inputs
    )
    return _select_targets(output, target)

def _select_targets(output: Tensor, target: TargetType) -> Tensor:
    # changed due to possible mask input in variable target
    # used to apply a mask which pixels in a segmentation should be returned
    # to the lrp
    if target is None:
        return output
    num_examples = output.shape[0]
    dims = len(output.shape)
    device = output.device
    if isinstance(target, (int, tuple)):
        return _verify_select_column(output, target)
    elif isinstance(target, torch.Tensor):
        if torch.numel(target) == 1 and isinstance(target.item(), int):
            return _verify_select_column(output, cast(int, target.item()))
        elif len(target.shape) == 1 and torch.numel(target) == num_examples:
            assert dims == 2, "Output must be 2D to select tensor of targets."
            return torch.gather(output, 1, target.reshape(len(output), 1))
        elif len(target.shape) > 1:
            assert target.shape == output.shape, f"Output {output.shape} and Mask {target.shape} (target) shape must match"
            #grad_outputs = torch.zeros(target.size(), dtype=torch.float32, requires_grad=False).to(device=target.device)
            #grad_outputs[target==True] = output[target==True]
            #return (grad_outputs)
            return output * target
        else:
            raise AssertionError(
                "Tensor target dimension %r is not valid. %r"
                % (target.shape, output.shape)
            )
    elif isinstance(target, list):
        assert len(target) == num_examples, "Target list length does not match output!"
        if isinstance(target[0], int):
            assert dims == 2, "Output must be 2D to select tensor of targets."
            return torch.gather(
                output, 1, torch.tensor(target, device=device).reshape(len(output), 1)
            )
        elif isinstance(target[0], tuple):
            return torch.stack(
                [
                    output[(i,) + cast(Tuple, targ_elem)]
                    for i, targ_elem in enumerate(target)
                ]
            )
        else:
            raise AssertionError("Target element type in list is not valid.")
    else:
        raise AssertionError("Target type %r is not valid." % target)
