#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.payloads import SparkPayload

import abc
import typing as tp
from spark.nn.interfaces.base import Interface
from spark.core.payloads import SpikeArray, FloatArray
from spark.nn.interfaces.base import Interface, InterfaceConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class OutputInterfaceOutput(tp.TypedDict):
    """
       OutputInterface model output spec.
    """
    signal: FloatArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class OutputInterfaceConfig(InterfaceConfig):
    """
        Abstract OutputInterface model configuration class.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=OutputInterfaceConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class OutputInterface(Interface, abc.ABC, tp.Generic[ConfigT]):
    """
        Abstract OutputInterface model.
    """
    
    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Main attributes
        super().__init__(config = config, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args: SpikeArray, **kwargs) -> dict[str, SparkPayload]:
        """
            Transform incomming spikes into a output signal.
        """
        pass

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################