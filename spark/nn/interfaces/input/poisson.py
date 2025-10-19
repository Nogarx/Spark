#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import jax
import dataclasses as dc
import spark.core.utils as utils
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.input.base import InputInterface, InputInterfaceConfig, InputInterfaceOutput

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class PoissonSpikerConfig(InputInterfaceConfig):
    """
        PoissonSpiker model configuration class.
    """

    max_freq: float = dc.field(
        default = 50.0, 
        metadata = {
            'units': 'Hz',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Maximum firing frequency of the spiker.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class PoissonSpiker(InputInterface):
    """
        Transforms a continuous signal to a spiking signal.
        This transformation assumes a very simple poisson neuron model without any type of adaptation or plasticity.

        Init:
            max_freq: float [Hz]

        Input:
            signal: FloatArray

        Output:
            spikes: SpikeArray
    """
    config: PoissonSpikerConfig

    def __init__(self, config: PoissonSpikerConfig | None = None, **kwargs):
        # Initialize super
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.max_freq = self.config.max_freq
        self._scale = self._dt * (self.max_freq / 1000)

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Initialize shapes
        self._shape = utils.validate_shape(input_specs['signal'].shape)

    def __call__(self, signal: FloatArray) -> InputInterfaceOutput:
        """
            Input interface operation.

            Input: 
                A FloatArray of values in the range [0,1].
            Output: 
                A SpikeArray of the same shape as the input.
        """
        spikes = (jax.random.uniform(self.get_rng_keys(1), shape=self._shape) < self._scale * signal.value).astype(self._dtype)
        return {
            'spikes': SpikeArray(spikes)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################