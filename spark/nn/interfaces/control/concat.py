#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax.numpy as jnp
import dataclasses as dc
import spark.core.utils as utils
from spark.core.specs import PortSpecs
from spark.core.registry import register_module, register_config
from spark.core.payloads import SparkPayload
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.control.base import ControlInterface, ControlInterfaceConfig, ControlInterfaceOutput, _build_signature_from_inputs

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ConcatConfig(ControlInterfaceConfig):
    """
        Concat configuration class.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class Concat(ControlInterface):
    """
        Combines several streams of inputs of the same type into a single stream.

        Init:
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

    def build(self, input_specs: dict[str, PortSpecs]) -> None:
        # Validate payloads types.
        payload_type = None
        for key, value in input_specs.items():
            payload_type = value.payload_type if payload_type is None else payload_type
            if payload_type != value.payload_type:
                raise TypeError(
                    f'Expected all payload types to be of same type \"{payload_type}\" '
                    f'but input spec \"{key}\" is of type "{value.payload_type}".'
                )
        self._payload_type = payload_type

    def _overwrite_call_signature(self, raw_args: tuple[SparkPayload], raw_kwargs: dict[str, SparkPayload]) -> None:
        # Create the new Signature object and assign it to the __call__ method
        self.__call__.__func__.__signature__ = _build_signature_from_inputs(raw_args, raw_kwargs)

    def __call__(self, **inputs: SparkPayload) -> ControlInterfaceOutput:
        """
            Merge all input streams into a single data output stream.
        """
        # Control flow operation
        return {
            'output': self._payload_type(jnp.concatenate([x.value.reshape(-1) for x in inputs.values()]))
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ConcatReshapeConfig(ConcatConfig):
    """
        ConcatReshape configuration class.
    """

    reshape: tuple[int, ...] = dc.field(
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
            reshape: tuple[int, ...]
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
        self.reshape = utils.validate_shape(self.config.reshape)

    def build(self, input_specs: dict[str, PortSpecs]) -> None:
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
            jnp.concatenate([jnp.zeros(s.shape).reshape(-1) for s in input_specs.values()]).reshape(self.reshape)
        except:
            raise ValueError(f'Shapes {[s.shape for s in input_specs.values()]} are not broadcastable to {self.reshape}')

    def _overwrite_call_signature(self, raw_args: tuple[SparkPayload], raw_kwargs: dict[str, SparkPayload]) -> None:
        # Create the new Signature object and assign it to the __call__ method
        self.__call__.__func__.__signature__ = _build_signature_from_inputs(raw_args, raw_kwargs)

    def __call__(self, **inputs: SparkPayload) -> ControlInterfaceOutput:
        """
            Merge all input streams into a single data output stream. Output stream is reshape to match the pre-specified shape.
        """
        # Control flow operation
        return {
            'output': self.payload_type(jnp.concatenate([x.value.reshape(-1) for x in inputs.values()]).reshape(self.reshape))
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################