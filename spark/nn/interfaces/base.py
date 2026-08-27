#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.payloads import SparkPayload
    
import abc
import typing as tp
from spark.core.module import SparkModule
from spark.core.config import DefaultSparkConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InterfaceOutput(tp.TypedDict):
    """
        Output ports of an interface.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InterfaceConfig(DefaultSparkConfig):
    """
        Base configuration for interfaces.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=InterfaceConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Interface(SparkModule, abc.ABC, tp.Generic[ConfigT]):
    """
        Base class for interfaces.

        An interface sits between a network and something that is not a network. Unlike a
        `Component`, it holds no neuronal state: it converts, routes or summarizes payloads.

        Parameters
        ----------
        config : InterfaceConfig
            Model configuration. Its fields may also be given as keyword arguments.

        See Also
        --------
        InputInterface : Turns an external signal into spikes.
        OutputInterface : Turns spikes into a continuous signal.
        ControlInterface : Routes and combines payloads inside a network.
    """
    config: ConfigT

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Main attributes
        super().__init__(config = config, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args: SparkPayload, **kwargs) -> InterfaceOutput:
        """
            Runs the interface operation.

            Parameters
            ----------
            *args : SparkPayload
                Inputs, as declared by the concrete interface.
            **kwargs
                Inputs, as declared by the concrete interface.

            Returns
            -------
            InterfaceOutput
                Dictionary of output ports, as declared by the concrete interface.
        """
        pass
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################