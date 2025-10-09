#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp
import dataclasses as dc
from math import prod
from spark.core.specs import InputSpec
from spark.core.variables import Constant
from spark.core.shape import Shape, Shape
from spark.core.registry import register_module, register_config
from spark.core.payloads import SparkPayload
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.control.base import ControlInterface, ControlInterfaceConfig, ControlInterfaceOutput

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

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Initialize shapes
        input_shape = Shape(input_specs['inputs'].shape)
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

    def __call__(self, inputs: SparkPayload) -> ControlInterfaceOutput:
        """
            Sub/Super-sample the input stream to get the pre-specified number of samples.
        """
        # Sample
        sample = type(inputs)(inputs.value.reshape(-1)[self.indices])
        return {
            'output': sample
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################