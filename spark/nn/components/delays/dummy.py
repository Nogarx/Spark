#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import jax.numpy as jnp
from math import prod
from spark.core.payloads import SpikeArray
from spark.core.variables import Variable
from spark.core.shape import Shape
from spark.core.registry import register_module, register_config
from spark.nn.components.delays.base import Delays, DelaysOutput, DelaysConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class DummyDelaysConfig(DelaysConfig):
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class DummyDelays(Delays):
    """
        A dummy delay that is equivalent to no delay at all, it simply forwards its input.
        This is a convinience module, that should be avoided whenever is possible and its 
        only purpose is to simplify some scenarios.

        Init:

        Input:
            in_spikes: SpikeArray
            
        Output:
            out_spikes: SpikeArray
    """
    config: DummyDelaysConfig

    def __init__(self, config: DummyDelaysConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize shapes
        self._shape = Shape(input_specs['in_spikes'].shape)
        self._units = prod(self._shape)
        # Initialize varibles
        self.spikes = Variable(jnp.zeros(self._shape, self._dtype), dtype=self._dtype)

    def reset(self) -> None:
        """
            Resets component state.
        """
        self.spikes.value = jnp.zeros(self._shape, self._dtype)

    def _push(self, in_spikes: SpikeArray) -> None:
        """
            Push operation.
        """
        self.spikes.value = in_spikes.value

    def _gather(self,) -> SpikeArray:
        """
            Gather operation.
        """
        return SpikeArray(self.spikes.value)

    # Override call, no need to store anything. Push and Gather are defined just for the sake of completion.
    def __call__(self, in_spikes: SpikeArray) -> DelaysOutput:
        return {
            'out_spikes': in_spikes,
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################