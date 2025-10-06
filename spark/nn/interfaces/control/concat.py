#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax.numpy as jnp
import dataclasses as dc
from spark.core.specs import InputSpec
from spark.core.registry import register_module, register_config
from spark.core.payloads import SparkPayload
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.core.shape import Shape
from spark.nn.interfaces.control.base import ControlInterface, ControlInterfaceConfig, ControlInterfaceOutput

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ConcatConfig(ControlInterfaceConfig):
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
class Concat(ControlInterface):
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

    def __init__(self, config: ConcatConfig | None = None, **kwargs):
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

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
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
class ConcatReshape(ControlInterface):
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

    def __init__(self, config: ConcatReshapeConfig | None = None, **kwargs):
		# Initialize super.
        super().__init__(config=config, **kwargs)
        # Intialize variables.
        self.reshape = Shape(self.config.reshape)
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

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################