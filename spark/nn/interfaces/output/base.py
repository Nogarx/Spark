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
        Output ports of an output interface.

        Attributes
        ----------
        signal : FloatArray
            The decoded signal.
    """
    signal: FloatArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class OutputInterfaceConfig(InterfaceConfig):
    """
        Base configuration for output interfaces.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=OutputInterfaceConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class OutputInterface(Interface, abc.ABC, tp.Generic[ConfigT]):
    """
        Base class for output interfaces.

        An output interface decodes spikes into a continuous signal, which is what lets something
        that is not a network read what a network produced.

        Parameters
        ----------
        config : OutputInterfaceConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        spikes : SpikeArray
            Spikes to decode.

        Output Ports
        ------------
        signal : FloatArray
            The decoded signal.

        See Also
        --------
        ExponentialIntegrator : Exponential filter of the incoming spikes.
    """
    
    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Main attributes
        super().__init__(config = config, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args: SpikeArray, **kwargs) -> dict[str, SparkPayload]:
        """
            Decodes the incoming spikes into a signal.

            Parameters
            ----------
            *args : SpikeArray
                Inputs, as declared by the concrete interface.
            **kwargs
                Inputs, as declared by the concrete interface.

            Returns
            -------
            dict of str to SparkPayload
                Dictionary with one entry, ``signal``.
        """
        pass

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################