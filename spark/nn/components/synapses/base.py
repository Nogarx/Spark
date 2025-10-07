#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import typing as tp
from spark.nn.components.base import Component, ComponentConfig
from spark.core.payloads import SpikeArray, CurrentArray, FloatArray

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic Soma output contract.
class SynanpsesOutput(tp.TypedDict):
    currents: CurrentArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SynanpsesConfig(ComponentConfig):
    pass
ConfigT = tp.TypeVar("ConfigT", bound=SynanpsesConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Synanpses(Component, tp.Generic[ConfigT]):
    """
        Abstract synapse model.

        Init:

        Input:
            spikes: SpikeArray
            
        Output:
            currents: CurrentArray
    """

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)

    @abc.abstractmethod
    def get_kernel(self,) -> FloatArray:
        pass

    @abc.abstractmethod
    def set_kernel(self, new_kernel: FloatArray) -> None:
        pass

    @abc.abstractmethod
    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        pass

    def __call__(self, spikes: SpikeArray) -> SynanpsesOutput:
        """
            Compute synanpse's currents.
        """
        return {
            'currents': self._dot(spikes)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################