#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import typing as tp
from spark.core.payloads import SpikeArray, IntegerArray
from spark.nn.components.base import Component, ComponentConfig
from spark.core.decorators import spark_property
from spark.core.variables import Variable

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class DelaysOutput(tp.TypedDict):
    """
       Generic delay model output spec.
    """
    out_spikes: SpikeArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class DelaysConfig(ComponentConfig):
    """
       Base synaptic delay configuration class.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=DelaysConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Delays(Component, tp.Generic[ConfigT]):
    """
        Abstract synaptic delay model.
    """
    _kernel: Variable

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)
    
    @spark_property
    def kernel(self,) -> IntegerArray:
        return IntegerArray(self._kernel.value)

    @kernel.setter
    def kernel(self, new_kernel: IntegerArray) -> None:
        self._kernel.value = new_kernel.value

    @abc.abstractmethod
    def _push(self, spikes: SpikeArray) -> None:
        """
            Push operation.
        """
        pass

    @abc.abstractmethod
    def _gather(self,) -> SpikeArray:
        """
            Gather operation.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
            Resets component state.
        """
        pass

    @abc.abstractmethod
    def __call__(self, in_spikes: SpikeArray) -> DelaysOutput:
        pass

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################