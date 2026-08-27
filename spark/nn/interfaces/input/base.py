#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.payloads import SparkPayload

import abc
import typing as tp
from spark.core.payloads import SpikeArray
from spark.nn.interfaces.base import Interface, InterfaceConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InputInterfaceOutput(tp.TypedDict):
    """
        Output ports of an input interface.

        Attributes
        ----------
        spikes : SpikeArray
            The encoded signal.
    """
    spikes: SpikeArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InputInterfaceConfig(InterfaceConfig):
    """
        Base configuration for input interfaces.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=InputInterfaceConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InputInterface(Interface, abc.ABC, tp.Generic[ConfigT]):
    """
        Base class for input interfaces.

        An input interface encodes a continuous signal as spikes, which is what lets a network
        read data that did not come from a network.

        Parameters
        ----------
        config : InputInterfaceConfig
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        signal : FloatArray
            Value to encode.

        Output Ports
        ------------
        spikes : SpikeArray
            The encoded signal.

        See Also
        --------
        PoissonSpiker : Stochastic rate encoding.
        LinearSpiker : Deterministic rate encoding.
    """
    config: ConfigT

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Main attributes
        super().__init__(config = config, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args: SparkPayload, **kwargs) -> InputInterfaceOutput:
        """
            Encodes the signal as spikes.

            Parameters
            ----------
            *args : SparkPayload
                Inputs, as declared by the concrete interface.
            **kwargs
                Inputs, as declared by the concrete interface.

            Returns
            -------
            InputInterfaceOutput
                Dictionary with one entry, ``spikes``.
        """
        pass

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################