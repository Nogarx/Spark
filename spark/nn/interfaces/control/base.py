#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import inspect
import typing as tp
from spark.core.payloads import SparkPayload
from spark.nn.interfaces.base import Interface, InterfaceConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _build_signature_from_inputs(raw_kwargs: dict[str, SparkPayload]) -> None:
    params = [inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    # Add inputs incrementally
    kwargs_names = set()
    kw_params = []
    for key, value in raw_kwargs.items():
        # Append param
        kw_params.append(inspect.Parameter(name=key, kind=inspect.Parameter.KEYWORD_ONLY, annotation=type(value)))
        kwargs_names.add(key)
    # Add raw_args
    key_idx = 0
    key = f'input_{key_idx}'
    #for value in raw_args:
    #    # Get next available key
    #    while key in kwargs_names:
    #        key_idx += 1
    #        key = f'input_{key_idx}'
    #    # Append param
    #    params.append(inspect.Parameter(name=key, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=type(value)))
    #    kwargs_names.add(key)
    # Combine args with kwargs
    params = params + kw_params
    return inspect.Signature(params)

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