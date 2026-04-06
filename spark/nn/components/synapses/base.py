#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import typing as tp
from spark.nn.components.base import Component, ComponentConfig
from spark.core.variables import Variable
from spark.core.payloads import SpikeArray, CurrentArray, FloatArray
from spark.core.decorators import spark_property

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class SynanpsesOutput(tp.TypedDict):
    """
       Generic synapses model output spec.
    """
    currents: CurrentArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SynanpsesConfig(ComponentConfig):
    """
       Abstract synapse model configuration class.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=SynanpsesConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Synapses(Component, tp.Generic[ConfigT]):
    """
        Abstract synapse model.

        Note that we require the kernel entries to be in pA for numerical stability, since most of the time we want to run in half-precision.
        However somas expect the current in nA so we need to rescale the output.

        Init:

        Input:
            spikes: SpikeArray
            
        Output:
            currents: CurrentArray
    """
    _kernel: Variable

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)

    @spark_property
    def kernel(self,) -> FloatArray:
        return FloatArray(self._kernel.value)

    @kernel.setter
    def kernel(self, new_kernel: FloatArray) -> None:
        self._kernel.value = new_kernel.value

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