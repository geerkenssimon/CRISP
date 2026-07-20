# built-in modules
import copy
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

# third-party modules
import torch
import torch.nn as nn

from captum._utils.common import _format_tensor_into_tuples
from captum._utils.typing import Module, Tensor

def _get_device(i):
    dev = None
    while dev is None:
        try:
            return i.device
        except:
            dev = _get_device(i[0])
    return dev

class PropagationRule(ABC):
    """
    Base class for all propagation rule classes, also called Z-Rule.
    STABILITY_FACTOR is used to assure that no zero divison occurs.
    """

    STABILITY_FACTOR = 1e-9

    def forward_hook(self, module, inputs, outputs):
        """Register backward hooks on input and output
        tensors of linear layers in the model."""
        self.module = module
        inputs = _format_tensor_into_tuples(inputs)
        self._has_single_input = len(inputs) == 1
        self._handle_input_hooks = []
        for input in inputs:
            if not hasattr(input, "hook_registered"):
                input_hook = self._create_backward_hook_input(input.data)
                self._handle_input_hooks.append(input.register_hook(input_hook))
                input.hook_registered = True
        output_hook = self._create_backward_hook_output(outputs.data)
        self._handle_output_hook = outputs.register_hook(output_hook)
        return outputs.clone()

    @staticmethod
    def backward_hook_activation(module, grad_input, grad_output):
        """Backward hook to propagate relevance over non-linear activations."""
        # replace_out is set in _backward_hook_input, this is necessary
        # due to 2 tensor hooks on the same tensor
        if hasattr(grad_output, "replace_out"):
            hook_out = grad_output.replace_out
            del grad_output.replace_out
            return hook_out
        return grad_output

    def _create_backward_hook_input(self, inputs):
        def _backward_hook_input(grad):
            relevance = grad * inputs
            device = grad.device
            if self._has_single_input:
                self.relevance_input[device] = relevance.data
            else:
                self.relevance_input[device].append(relevance.data)

            # replace_out is needed since two hooks are set on the same tensor
            # The output of this hook is needed in backward_hook_activation
            grad.replace_out = relevance
            return relevance

        return _backward_hook_input

    def _create_backward_hook_output(self, outputs):
        def _backward_hook_output(grad):
            sign = torch.sign(outputs)
            sign[sign == 0] = 1
            relevance = grad / (outputs + sign * self.STABILITY_FACTOR)
            self.relevance = relevance
            self.relevance_output[grad.device] = grad.data
            return relevance

        return _backward_hook_output

    def forward_hook_weights(self, module, inputs, outputs):
        """Save initial activations a_j before modules are changed"""
        device = _get_device(inputs)
        #device = None
        #while device is None:
        #    try:
        #        device = inputs[0].device
        #    except:
        #        inputs = inputs[0]
        #device = inputs[0].device if isinstance(inputs, tuple) else inputs.device
        if hasattr(module, "activations") and device in module.activations:
            raise RuntimeError(
                "Module {} is being used more than once in the network, which "
                "is not supported by LRP. "
                "Please ensure that module is being used only once in the "
                "network.".format(module)
            )
        module.activations[device] = ()
        def traverse(lst):
            for item in lst:
                if isinstance(item, torch.Tensor):
                    module.activations[device] += tuple([item])
                elif isinstance(item, list):  # If item is a nested list, recurse
                    traverse(item)
        traverse(inputs)
        #module.activations[device] = tuple(input.data for input in inputs)
        module.outputs[device] = tuple(output.data for output in outputs if not type(output) == list)
        self._manipulate_weights(module, inputs, outputs)

    @abstractmethod
    def _manipulate_weights(self, module, inputs, outputs):
        raise NotImplementedError

    def forward_pre_hook_activations(self, module, inputs):
        """Pass initial activations to graph generation pass"""
        device = _get_device(inputs)
        _inputs = []
        def traverse(lst):
            for item in lst:
                if isinstance(item, torch.Tensor):
                    _inputs.append(item)
                elif isinstance(item, list):  # If item is a nested list, recurse
                    traverse(item)
        traverse(inputs)
        #device = inputs[0].device if isinstance(inputs, tuple) else inputs.device
        for input, activation in zip(_inputs, module.activations[device]):
            input.data = activation
        return inputs


class EpsilonRule(PropagationRule):
    """
    Rule for relevance propagation using a small value of epsilon
    to avoid numerical instabilities and remove noise.

    Use for middle layers.

    Args:
        epsilon (integer, float): Value by which is added to the
        discriminator during propagation.
    """

    def __init__(self, epsilon=0.25) -> None:
        self.STABILITY_FACTOR = epsilon

    def _manipulate_weights(self, module, inputs, outputs):
        pass


class GammaRule(PropagationRule):
    """
    Gamma rule for relevance propagation, gives more importance to
    positive relevance.

    Use for lower layers.

    Args:
        gamma (float): The gamma parameter determines by how much
        the positive relevance is increased.
    """

    def __init__(self, gamma=0.25, set_bias_to_zero=False) -> None:
        self.gamma = gamma
        self.set_bias_to_zero = set_bias_to_zero

    def _manipulate_weights(self, module, inputs, outputs):
        if hasattr(module, "weight"):
            module.weight.data = (
                module.weight.data + self.gamma * copy.deepcopy(module.weight.data.clamp(min=0))
            )
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)


class Alpha1_Beta0_Rule(PropagationRule):
    """
    Alpha1_Beta0 rule for relevance backpropagation, also known
    as Deep-Taylor. Only positive relevance is propagated, resulting
    in stable results, therefore recommended as the initial choice.
    Warning: Does not work for BatchNorm modules because weight and bias
    are defined differently.
    Use for lower layers.
    """

    def __init__(self, set_bias_to_zero=False) -> None:
        self.set_bias_to_zero = set_bias_to_zero

    def _manipulate_weights(self, module, inputs, outputs):
        if hasattr(module, "weight"):
            module.weight.data = module.weight.data.clamp(min=0)
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)


class WW_Rule(PropagationRule):
    def __init__(self) -> None:
        self._input_shapes: Tuple[torch.Size, ...] = tuple()
        self._denominator = torch.Tensor()

    def forward_hook(
        self, module: Module, inputs: Tuple[Tensor, ...], outputs: Tensor
    ) -> Tensor:
        self._compute_denominator(module, inputs)
        return super().forward_hook(module, inputs, outputs)

    def _compute_denominator(self, module: Module, inputs: Tuple[Tensor, ...]) -> None:
        input_shapes = tuple(x.shape[1:] for x in inputs)

        if input_shapes != self._input_shapes:
            self._input_shapes = input_shapes
            with torch.no_grad():
                self._denominator = module.forward(
                    *tuple(
                        torch.ones(input_shape).unsqueeze(dim=0).to(inputs[0].device)
                        for input_shape in self._input_shapes
                    )
                )
                self._denominator += self.STABILITY_FACTOR

    def _create_backward_hook_input(self, input_: Tensor) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_input(grad: Tensor,) -> None:
            pass

        return _backward_hook_input

    def _create_backward_hook_output(self, output: Tensor) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_output(grad: Tensor) -> Tensor:
            relevance = grad / torch.cat(
                tuple(self._denominator for _ in range(output.shape[0]))
            )
            return relevance

        return _backward_hook_output

    def _manipulate_weights(self, module: Module, inputs: Tuple[Tensor, ...], outputs: Tensor,) -> None:
        if hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)
        if hasattr(module, "weight"):
            module.weight.data = module.weight.data ** 2

class FlatRule(WW_Rule):
    def __init__(self):
        super().__init__()

    def _manipulate_weights(self, module: Module, inputs: Tuple[Tensor, ...], outputs: Tensor,) -> None:
        if hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)
        if hasattr(module, "weight"):
            module.weight.data = torch.ones_like(module.weight.data)



class zBRule(PropagationRule):
    def __init__(
        self,
        pixel_values:Tuple[float] = None,
        set_bias_to_zero: bool = False,
    ) -> None:
        """
        Args:
            lower_bound (Union[int, float], optional): Lower bound value for input
            features. Defaults to -1.0.
            upper_bound (Union[int, float], optional): Upper bound for input features.
            Defaults to 1.0.
            set_bias_to_zero (bool, optional): Parameter for setting bias to
            zero in relevance computations.
            Defaults to False.
        """
        self.upper_bound = pixel_values[1]
        self.lower_bound = pixel_values[0]
        self.set_bias_to_zero = set_bias_to_zero

        self._lower_bound_tensor = torch.Tensor()
        self._upper_bound_tensor = torch.Tensor()

        self._module_pos: Module = None
        self._module_neg: Module = None
        self._bias_contrib: Tensor = None

        self._denominator_bound_contribution = torch.Tensor()
        self._input_shapes: Tuple[torch.Size, ...] = tuple()

    def forward_hook(self, module: Module, inputs: Tuple[Tensor, ...], outputs: Tensor) -> Tensor:
        r"""
        Register backward hooks on input and output
        tensors of linear layers in the model.
        """
        if not hasattr(module, "weight"):
            raise RuntimeError(
                f"{self.__class__.__name__} assigned to module without weights:"
                + "{module}. This rule only supports modules with weight."
            )
        inputs = _format_tensor_into_tuples(inputs)
        self._handle_input_hooks = list()
        for input_index, input_ in enumerate(inputs):
            if not hasattr(input_, "hook_registered"):
                input_hook = self._create_backward_hook_input(input_.data, input_index)
                handle = input_.register_hook(input_hook)
                self._handle_input_hooks.append(handle)
                input_.hook_registered = True
        output_hook = self._create_backward_hook_output(outputs.data)
        self._handle_output_hook = outputs.register_hook(output_hook)

        self._create_auxiliary_quantities(inputs)

        return outputs.clone()

    def _create_backward_hook_input(self, input_: Tensor, input_index: int) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_input(grad: Tensor) -> Tensor:
            if hasattr(self, 'negative_weight_contraction'):            
                relevance = (input_ - self._lower_bound_tensor[input_index]) * grad
                relevance += (
                    input_ - self._upper_bound_tensor[input_index]
                ) * self.negative_weight_contraction[input_index]
                return relevance
            else:
                return input_ * grad

        return _backward_hook_input

    def _create_backward_hook_output(self, output: Tensor) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_output(grad: Tensor) -> Tensor:
            denominator = self.og_outputs[0] - self._denominator_bound_contribution
            if self.set_bias_to_zero and self._bias_contrib is not None:
                denominator -= torch.cat(
                    tuple(self._bias_contrib for _ in range(output.shape[0]))
                )
            denominator += self.STABILITY_FACTOR

            rescaled_relevance = grad / denominator
            self.negative_weight_contraction = torch.autograd.grad(
                outputs=self.upper_bound_contrib,
                inputs=self._upper_bound_tensor,
                grad_outputs=rescaled_relevance,
                retain_graph=True,
            )
            return rescaled_relevance

        return _backward_hook_output

    def _create_auxiliary_quantities(self, inputs: Tuple[Tensor, ...]) -> None:
        r"""
        Computes the l w^+ + h w^- term for the denominator.
        """
        input_shapes = tuple(x.shape for x in inputs)

        if input_shapes != self._input_shapes:
            self._input_shapes = input_shapes

            with torch.autograd.set_grad_enabled(True):
                self._upper_bound_tensor = tuple(
                    torch.full(
                        input_shape,
                        self.upper_bound,
                        dtype=torch.float,
                        requires_grad=True,
                    ).to(inputs[0].device)
                    for input_shape in self._input_shapes
                )
                self.module.weight.data = copy.deepcopy(self._module_neg_weights)
                self.upper_bound_contrib = self.module.forward(
                    *self._upper_bound_tensor
                )
                #self.upper_bound_contrib = self._module_neg.forward(
                #    *self._upper_bound_tensor
                #)

            with torch.no_grad():
                self.module.weight.data = copy.deepcopy(self._module_pos_weights)
                self._lower_bound_tensor = tuple(
                    torch.full(input_shape, self.lower_bound, dtype=torch.float).to(inputs[0].device)
                    for input_shape in self._input_shapes
                )
                self._denominator_bound_contribution = (
                    self.module.forward(*self._lower_bound_tensor)
                    + self.upper_bound_contrib
                )
                #self._denominator_bound_contribution = (
                #    self._module_pos.forward(*self._lower_bound_tensor)
                #    + self.upper_bound_contrib
                #)
            self.module.weight.data += copy.deepcopy(self._module_neg_weights)

    def forward_hook_weights(self, module: Module, inputs: Tuple[Tensor, ...], outputs: Tensor,) -> None:
        self.og_outputs = outputs.detach()
        super().forward_hook_weights(module, inputs, outputs)

    def _separate_weights_by_sign(self, module: Module) -> None:
        self.module = module
        if hasattr(module, "weight"):
            self._module_neg_weights = copy.deepcopy(module.weight.data)
            self._module_neg_weights = self._module_neg_weights.clamp(max=0.0)
            self._module_pos_weights = copy.deepcopy(module.weight.data)
            self._module_pos_weights = self._module_pos_weights.clamp(min=0.0)

    def _manipulate_weights(self, module: Module, inputs: Tuple[Tensor, ...], outputs: Tensor,) -> None:
        if hasattr(module, "bias"):
            if module.bias is not None:
                if self.set_bias_to_zero and self._bias_contrib is None:
                    with torch.no_grad():
                        self._bias_contrib = module.forward(
                            *(
                                torch.zeros(input_.shape[1:]).unsqueeze(dim=0).to(inputs[0].device)
                                for input_ in inputs
                            )
                        )
                module.bias.data = torch.zeros_like(module.bias.data)

        self._separate_weights_by_sign(module)
    
class Alpha_Beta_Rule(zBRule):
    STABILITY_FACTOR = 1e-16
    def __init__(self, alpha: float = 1.0, beta: float = None, set_bias_to_zero: bool = False) -> None:
        r"""
        Args:
            alpha (float, optional): Alpha parameter of alpha beta rule.
            Defaults to 1.

            set_bias_to_zero (bool, optional): Parameter for setting bias to
            zero in relevance computations.
            Defaults to False.
        """
        self.alpha = alpha
        if beta is not None:
            self.beta = beta
        else:
            self.beta = 1.0 - self.alpha
        self.set_bias_to_zero = set_bias_to_zero

        #self._module_pos: Optional[Module] = None
        #self._module_neg: Optional[Module] = None
        self._bias_contrib: Tensor = None

    def _create_backward_hook_input(
        self, input_: Tensor, input_index: int
    ) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_input(grad: Tensor) -> Tensor:
            if hasattr(self, 'out'):
                return self.out[input_index]
            else:
                return input_ * grad

        return _backward_hook_input

    def _create_backward_hook_output(
        self, output: Tensor
    ) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_output(grad: Tensor) -> None:
            out = self._compute_signed_contributions(
                grad, self.inputs_pos, self.inputs_neg, +1
            )

            if self.beta:
                out_beta = tuple(
                    self.beta * rel
                    for rel in self._compute_signed_contributions(
                        grad, self.inputs_neg, self.inputs_pos, -1
                    )
                )
                out = tuple(self.alpha * x + y for x, y in zip(out, out_beta))
            self.out = out

        return _backward_hook_output

    def _compute_signed_contributions(
        self,
        grad: Tensor,
        mod_pos_in: Tuple[Tensor, ...],
        mod_neg_in: Tuple[Tensor, ...],
        sign: int,
    ) -> Tuple[Tensor, ...]:
        r"""
        if mod_pos_in is the positive part of the inputs and mod_neg_in the negative
        this computes the alpha part
        if mod_pos_in is the negative part of the inputs and mod_neg_in the positive
        this computes the beta part
        """
        with torch.autograd.set_grad_enabled(True):
            self.module.weight.data = copy.deepcopy(self._module_pos_weights)
            #self.module.weight.data = self._module_pos_weights
            mod_pos_out = self.module.forward(*mod_pos_in)
            self.module.weight.data = copy.deepcopy(self._module_neg_weights)
            #self.module.weight.data = self._module_neg_weights
            mod_neg_out = self.module.forward(*mod_neg_in)
            self.module.weight.data += copy.deepcopy(self._module_pos_weights)
            #self.module.weight.data += self._module_pos_weights

            denominator = mod_pos_out + mod_neg_out
            if not self.set_bias_to_zero and self._bias_contrib is not None:
                if sign == 1:
                    denominator += torch.cat(
                        tuple(self._bias_contrib for _ in range(denominator.shape[0]))
                    ).clamp(min=0)
                else:
                    denominator += torch.cat(
                        tuple(self._bias_contrib for _ in range(denominator.shape[0]))
                    ).clamp(max=0)

            # this might be unneccessary, simply adding epsilon may be enough
            denominator += sign * (torch.eq(denominator, 0.0)) * self.STABILITY_FACTOR

            rescaled_relevance = grad / denominator

            self.module.weight.data = copy.deepcopy(self._module_pos_weights)
            # getting contractions with transposed Jacobian
            positive_weight_contraction = torch.autograd.grad(
                outputs=mod_pos_out, inputs=mod_pos_in, grad_outputs=rescaled_relevance
            )
            self.module.weight.data = copy.deepcopy(self._module_neg_weights)
            negative_weight_contraction = torch.autograd.grad(
                outputs=mod_neg_out, inputs=mod_neg_in, grad_outputs=rescaled_relevance
            )
            
            self.module.weight.data += copy.deepcopy(self._module_pos_weights)
            out = tuple(
                (pos_in * jac_pos) + (neg_in * jac_neg)
                for jac_pos, pos_in, jac_neg, neg_in in zip(
                    positive_weight_contraction,
                    mod_pos_in,
                    negative_weight_contraction,
                    mod_neg_in,
                )
            )

        return out

    def _create_auxiliary_quantities(self, inputs: Tuple[Tensor, ...]) -> None:
        self.inputs_pos = tuple(input_.data.clamp(min=0) for input_ in inputs)
        self.inputs_neg = tuple(input_.data.clamp(max=0) for input_ in inputs)
        #self.inputs_neg = tuple(input_.data.clamp(min=0) for input_ in inputs)
        #self.inputs_pos = tuple(input_.data.clamp(max=0) for input_ in inputs)

        for input_pos, input_neg in zip(self.inputs_pos, self.inputs_neg):
            input_pos.requires_grad_(True)
            input_neg.requires_grad_(True)

    def _manipulate_weights(
        self,
        module: Module,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
    ) -> None:
        if hasattr(module, "bias"):
            if module.bias is not None:
                if not self.set_bias_to_zero and self._bias_contrib is None:
                    with torch.no_grad():
                        self._bias_contrib = module.forward(
                            *(
                                torch.zeros(input_.shape[1:], device=inputs[0].device).unsqueeze(dim=0)
                                for input_ in inputs
                            )
                        )
                module.bias.data = torch.zeros_like(module.bias.data)

        self._separate_weights_by_sign(module)


class IdentityRule(EpsilonRule):
    """
    Identity rule for skipping layer manipulation and propagating the
    relevance over a layer. Only valid for modules with same dimensions for
    inputs and outputs.

    Can be used for BatchNorm2D.
    """

    def __init__(self):
        self.STABILITY_FACTOR = 0

    def _create_backward_hook_input(self, inputs):
        def _backward_hook_input(grad):
            #return grad
            return self.relevance_output[grad.device]

        return _backward_hook_input


class UpsampleRule(PropagationRule):
    def __init__(self):
        self.STABILITY_FACTOR = 0

    def forward_hook(self, module, inputs: torch.Tensor, outputs: torch.Tensor):
        self.scale_factor = module.scale_factor
        return super().forward_hook(module, inputs, outputs)

    def _create_backward_hook_output(self, outputs):
        def _backward_hook_output(grad):
            sign = torch.sign(outputs)
            sign[sign == 0] = 1
            relevance = grad
            self.relevance_output[grad.device] = grad.data
            return relevance

        return _backward_hook_output
    
    def _manipulate_weights(self, module, inputs, outputs):
        pass

class ContainerLayerRule(PropagationRule):
    def __init__(self) -> None:
        self.STABILITY_FACTOR = 0

    def forward_hook(self, module:nn.Module, inputs:torch.Tensor, outputs:torch.Tensor):
        """Register backward hooks on input and output
        tensors of linear layers in the model."""
        self.module = module
        inputs = _format_tensor_into_tuples(inputs)
        self._has_single_input = len(inputs) == 1
        self._handle_input_hooks = []
    
    def _manipulate_weights(self, module, inputs, outputs):
        pass

    def _create_backward_hook_output(self, outputs):
        def _backward_hook_output(grad):
            sign = torch.sign(outputs)
            sign[sign == 0] = 1
            relevance = grad
            self.relevance_output[grad.device] = grad.data
            if hasattr(self.module, 'input_grad'):
                self.module.output_grad[grad.device] = grad.data
            return relevance

        return _backward_hook_output
    

class NoneRule(PropagationRule):
    def __init__(self) -> None:
        self.STABILITY_FACTOR = 0

    def forward_hook(self, module:nn.Module, inputs:torch.Tensor, outputs:torch.Tensor):
        """Register backward hooks on input and output
        tensors of linear layers in the model."""
        self.module = module
        inputs = _format_tensor_into_tuples(inputs)
        self._has_single_input = len(inputs) == 1
        self._handle_input_hooks = []
    
    def _manipulate_weights(self, module, inputs, outputs):
        pass

    def _create_backward_hook_output(self, outputs):
        def _backward_hook_output(grad):
            sign = torch.sign(outputs)
            sign[sign == 0] = 1
            relevance = grad
            self.relevance_output[grad.device] = grad.data
            if hasattr(self.module, 'input_grad'):
                self.module.output_grad[grad.device] = grad.data
            return relevance

        return _backward_hook_output
    

def _print_hook(module:nn.Module, inputs:torch.Tensor, outputs:torch.Tensor):
    print(module.name)