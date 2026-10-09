#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import typing as tp
from spark.core.payloads import SparkPayload
from spark.nn.interfaces.base import Interface, InterfaceConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ControlInterfaceOutput(tp.TypedDict):
    """
        Output ports of a control interface.

        Attributes
        ----------
        output : SparkPayload
            The result of the operation. Its type follows the inputs.
    """
    output: SparkPayload

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ControlInterfaceConfig(InterfaceConfig):
    """
        Base configuration for control interfaces.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=ControlInterfaceConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ControlInterface(Interface, abc.ABC, tp.Generic[ConfigT]):
    """
        Base class for control interfaces.

        A control interface moves and transforms payloads around a graph rather than modelling anything.

        Parameters
        ----------
        config : ControlInterfaceConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        **inputs : SparkPayload
            Named by the graph that wires them, not by the signature.

        Output Ports
        ------------
        output : SparkPayload
            Result of the operation. Its type follows the inputs.

        See Also
        --------
        Concat : Joins several inputs of one type along an axis.
        Sampler : Draws a subset of one input.
        SignalTrace : Exponentially decaying trace of an input.
    """
    config: ConfigT

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args: SparkPayload, **kwargs) -> ControlInterfaceOutput:
        """
            Runs the control operation.

            Parameters
            ----------
            *args : SparkPayload
                Inputs, named by the graph rather than by the signature.
            **kwargs
                Inputs, named by the graph rather than by the signature.

            Returns
            -------
            ControlInterfaceOutput
                Dictionary with one entry, ``output``.
        """
        pass

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################