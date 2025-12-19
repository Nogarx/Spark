#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs

import abc
import jax.numpy as jnp
import typing as tp
import spark.core.utils as utils
from spark.core.variables import Variable
from spark.nn.components.base import Component, ComponentConfig
from spark.core.payloads import SpikeArray, CurrentArray, PotentialArray

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class SomaOutput(tp.TypedDict):
    """
       Generic soma model output spec.
    """
    spikes: SpikeArray
    potentials: PotentialArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SomaConfig(ComponentConfig):
    """
        Abstract soma model configuration class.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=SomaConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Soma(Component, tp.Generic[ConfigT]):
    """
        Abstract soma model.
    """
    config: ConfigT

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)
    
    def build(self, input_specs: dict[str, PortSpecs]):
        # Initialize shapes
        self.units = utils.validate_shape(input_specs['current'].shape)
        # Initialize variables
        self.potential = Variable(jnp.zeros(self.units, dtype=self._dtype), dtype=self._dtype)

    def reset(self):
        """
            Resets neuron states to their initial values.
        """
        self.potential.value = jnp.zeros(self.units, dtype=self._dtype)

    @abc.abstractmethod
    def _update_states(self, current: CurrentArray) -> None:
        """
            Update neuron's states variables.
        """
        pass

    @abc.abstractmethod
    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        pass

    def __call__(self, current: CurrentArray) -> SomaOutput:
        """
            Update neuron's states and compute spikes.
        """
        self._update_states(current)
        return {
            'spikes': self._compute_spikes(), 
            'potentials': self.potential,
        }
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################