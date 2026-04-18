#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp
import dataclasses as dc
import spark.core.utils as utils
from math import prod
from spark.core.specs import PortSpecs
from spark.core.variables import Constant
from spark.core.registry import register_module, register_config
from spark.core.payloads import SparkPayload
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.control.base import ControlInterface, ControlInterfaceConfig, ControlInterfaceOutput, _build_signature_from_inputs

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class SamplerConfig(ControlInterfaceConfig):
    """
        Sampler configuration class.
    """
    
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
class Sampler(ControlInterface):
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

    def __init__(self, config: SamplerConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.sample_size = self.config.sample_size

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
        # Initialize shapes
        flat_shape = sum([prod(spec.shape) for spec in input_specs.values()])
        input_shape = utils.validate_shape((flat_shape,))
        # Initialize variables
        self._indices = Constant(
            jax.random.randint(
                self.get_rng_keys(1), 
                self.sample_size, 
                minval=0, 
                maxval=prod(input_shape)
            ), 
            dtype=jnp.uint32
        )

    @property
    def indices(self,) -> jax.Array:
        return self._indices.value

    def _overwrite_call_signature(self, raw_args: tuple[SparkPayload], raw_kwargs: dict[str, SparkPayload]) -> None:
        # Create the new Signature object and assign it to the __call__ method
        self.__call__.__func__.__signature__ = _build_signature_from_inputs(raw_args, raw_kwargs)

    def __call__(self, **inputs: SparkPayload) -> ControlInterfaceOutput:
        """
            Sub/Super-sample the input stream to get the pre-specified number of samples.
        """
        # Control flow operation
        return {
            'output': self._payload_type(
                jnp.concatenate([x.value.reshape(-1) for x in inputs.values()])[self.indices]
            )
        }
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################