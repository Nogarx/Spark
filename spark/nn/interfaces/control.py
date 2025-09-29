#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax
import jax.numpy as jnp
import dataclasses as dc
import typing as tp
from math import prod
from spark.core.specs import InputSpec, OutputSpec
from spark.core.variables import Constant
from spark.core.shape import bShape, Shape, normalize_shape
from spark.core.registry import register_module
from spark.core.payloads import SparkPayload
from spark.core.config import SparkConfig
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.base import Interface

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic OutputInterface output contract.
class ControlInterfaceOutput(tp.TypedDict):
    output: SparkPayload

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ControlFlowInterface(Interface, abc.ABC):
    """
        Abstract control flow model.
    """

    def __init__(self, config: SparkConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ConcatConfig(SparkConfig):
    num_inputs: int = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Dtype used for JAX dtype promotions.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class Concat(ControlFlowInterface):
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
    config: ConcatConfig

    def __init__(self, config: ConcatConfig = None, **kwargs):
		# Initialize super.
        super().__init__(config=config, **kwargs)
        # Intialize variables.
        self.num_inputs = self.config.num_inputs

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Validate payloads types.
        payload_type = None
        for key, value in input_specs.items():
            payload_type = value.payload_type if payload_type is None else payload_type
            if payload_type != value.payload_type:
                raise TypeError(
                    f'Expected all payload types to be of same type \"{payload_type}\" '
                    f'but input spec \"{key}\" is of type "{value.payload_type}".'
                )
        self.payload_type = payload_type

    def __call__(self, input: list[SparkPayload]) -> ControlInterfaceOutput:
        """
            Merge all input streams into a single data output stream.
        """
        # Control flow operation
        return {
            'output': self.payload_type(jnp.concatenate([x.value.reshape(-1) for x in input]))
        }
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ConcatReshapeConfig(ConcatConfig):
    reshape: Shape = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Target shape after the merge operation.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ConcatReshape(ControlFlowInterface):
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
    config: ConcatReshapeConfig

    def __init__(self, config: ConcatReshapeConfig = None, **kwargs):
		# Initialize super.
        super().__init__(config=config, **kwargs)
        # Intialize variables.
        self.reshape = normalize_shape(self.config.reshape)
        self.num_inputs = self.config.num_inputs

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Validate payloads types.
        payload_type = None
        for key, value in input_specs.items():
            payload_type = value.payload_type if payload_type is None else payload_type
            if payload_type != value.payload_type:
                raise TypeError(
                    f'Expected all payload types to be of same type \"{payload_type}\" '
                    f'but input spec \"{key}\" is of type "{value.payload_type}".'
                )
        self.payload_type = payload_type
        # Validate final shape.
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
            'output': self.payload_type(jnp.concatenate([x.value.reshape(-1) for x in inputs]).reshape(self.reshape))
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SamplerConfig(SparkConfig):
    sample_size: int = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Sample size to drawn from the population. May be larger than the population.',
        })

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
    config: SamplerConfig

    def __init__(self, config: SamplerConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.sample_size = self.config.sample_size

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