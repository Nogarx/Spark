#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax
import jax.numpy as jnp
from math import prod
from typing import TypedDict, Dict, List
from spark.core.specs import InputSpec, OutputSpec
from spark.core.variables import Constant
from spark.core.shape import bShape, Shape, normalize_shape
from spark.core.registry import register_module
from spark.core.payloads import SparkPayload
from spark.nn.interfaces.base import Interface

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic OutputInterface output contract.
class ControlInterfaceOutput(TypedDict):
    output: SparkPayload

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ControlFlowInterface(Interface, abc.ABC):
    """
        Abstract control flow model.
    """

    def __init__(self, **kwargs):
        # Initialize super.
        super().__init__(**kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class Merger(ControlFlowInterface):
    """
        Combines several streams of inputs of the same type into a single stream.

        Init:
            num_inputs: int
            payload_type: type[SparkPayload]
            
        Input:
            input: type[SparkPayload]
            
        Output:
            output: type[SparkPayload]
    """

    def __init__(self, num_inputs: int, payload_type: type[SparkPayload], **kwargs):
		# Initialize super.
        super().__init__(**kwargs)
        # Intialize variables.
        self.num_inputs = num_inputs
        self.set_fallback_payload_type(payload_type)

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Validate payloads types.
        for key, value in input_specs.items():
            if self._default_payload_type != value.payload_type:
                raise TypeError(f'Expected "payload" to be of same type "{self._default_payload_type}" '
                                f'but input spec "{key}" is of type "{value.payload_type}".')

    def __call__(self, input: list[SparkPayload]) -> ControlInterfaceOutput:
        """
            Merge all input streams into a single data output stream.
        """
        # Control flow operation
        return {
            'output': self._default_payload_type(jnp.concatenate([x.value.reshape(-1) for x in input]))
        }
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class MergerReshape(ControlFlowInterface):
    """
        Combines several streams of inputs of the same type into a single stream.

        Init:
            num_inputs: int
            reshape: Shape
            payload_type: type[SparkPayload]
            
        Input:
            input: type[SparkPayload]
            
        Output:
            output: type[SparkPayload]
    """

    def __init__(self, num_inputs: int, reshape: Shape, payload_type: type[SparkPayload], **kwargs):
		# Initialize super.
        super().__init__(**kwargs)
        # Intialize variables.
        self.reshape = normalize_shape(reshape)
        self.num_inputs = num_inputs
        self.set_fallback_payload_type(payload_type)

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Validate payloads types.
        if self._default_payload_type != input_specs['inputs'].payload_type:
            raise TypeError(f'Expected "payload" to be of same type "{self._default_payload_type}" '
                            f'but input spec "inputs" is of type "{input_specs['inputs'].payload_type}".')
        # Validate can reshape.
        try:
            jnp.concatenate([jnp.zeros(s).reshape(-1) for s in input_specs['inputs'].shape]).reshape(self.reshape)
        except:
            raise ValueError(f'Shapes {input_specs['inputs'].shape} are not broadcastable to {self.reshape}')


    def __call__(self, inputs: list[SparkPayload]) -> ControlInterfaceOutput:
        """
            Merge all input streams into a single data output stream. Output stream is reshape to match the pre-specified shape.
        """
        # Control flow operation
        return {
            'output': self._default_payload_type(jnp.concatenate([x.value.reshape(-1) for x in inputs]).reshape(self.reshape))
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class Sampler(ControlFlowInterface):
    """
        Sample a single input streams of inputs of the same type into a single stream.
        Indices are selected randomly and remain fixed.

        Init:
            sample_size: int
            
        Input:
            input: type[SparkPayload]
            
        Output:
            output: type[SparkPayload]
    """

    def __init__(self, sample_size: int, **kwargs):
        # Initialize super.
        super().__init__(**kwargs)
        # Initialize variables
        self.sample_size = sample_size

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Initialize shapes
        input_shape = normalize_shape(input_specs['input'].shape)
        # Initialize variables
        self._indices = Constant(jax.random.randint(self.get_rng_keys(1), 
                                                         self.sample_size, 
                                                         minval=0, 
                                                         maxval=prod(input_shape)), 
                                                         dtype=jnp.uint32)

    @property
    def indices(self,) -> jax.Array:
        return self._indices.value

    def __call__(self, input: SparkPayload) -> ControlInterfaceOutput:
        """
            Sub/Super-sample the input stream to get the pre-specified number of samples.
        """
        # Sample
        sample = type(input)(input.value.reshape(-1)[self.indices])
        return {
            'output': sample
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################