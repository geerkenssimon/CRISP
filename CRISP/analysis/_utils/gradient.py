# built-in modules
from typing import Callable, Tuple, Union, Any

# third-party modules
import torch
from torch import Tensor

from captum._utils.typing import (
    TargetType
)

# modules from current project
from CRISP.analysis._utils.common import _run_forward

def compute_gradients(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target: TargetType = None,
    additional_forward_args: Any = None,
) -> Tuple[Tensor, ...]:
    r"""
    Computes gradients of the output with respect to inputs for an
    arbitrary forward function.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        input:      Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        additional_forward_args: Additional input arguments that forward
                    function requires. It takes an empty tuple (no additional
                    arguments) if no additional arguments are required
    """
    with torch.autograd.set_grad_enabled(True):
        # runs forward pass
        outputs = _run_forward(forward_fn, inputs, target, additional_forward_args)
        
        return torch.autograd.grad(torch.unbind(outputs), inputs, grad_outputs=torch.unbind(outputs), allow_unused=True)
        #return torch.autograd.grad(torch.unbind(outputs), inputs)
