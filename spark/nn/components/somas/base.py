#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import abc
import jax.numpy as jnp
import typing as tp
from spark.core.shape import Shape
from spark.core.variables import Variable
from spark.nn.components.base import Component, ComponentConfig
from spark.core.payloads import SpikeArray, CurrentArray, PotentialArray

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic Soma output contract.
class SomaOutput(tp.TypedDict):
    spikes: SpikeArray
    potentials: PotentialArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SomaConfig(ComponentConfig):
    pass
ConfigT = tp.TypeVar("ConfigT", bound=SomaConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Soma(Component, tp.Generic[ConfigT]):
    """
        Abstract soma model.
    """
    config: ConfigT

    # Type hints
    _potential: Variable
    _shape: Shape

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)

    @property
    def potential(self,) -> PotentialArray:
        """
            Returns the current soma potential.
        """
        return PotentialArray(self._potential.value)
    
    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize shapes
        self._shape = Shape(input_specs['current'].shape)
        # Initialize variables
        self._potential = Variable(jnp.zeros(self._shape, dtype=self._dtype), dtype=self._dtype)

    @abc.abstractmethod
    def reset(self):
        """
            Resets neuron states to their initial values.
        """
        pass

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