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

def _build_signature_from_inputs(raw_args: tuple[SparkPayload], raw_kwargs: dict[str, SparkPayload]) -> None:
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
    for value in raw_args:
        # Get next available key
        while key in kwargs_names:
            key_idx += 1
            key = f'input_{key_idx}'
        # Append param
        params.append(inspect.Parameter(name=key, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=type(value)))
        kwargs_names.add(key)
    # Combine args with kwargs
    params = params + kw_params
    return inspect.Signature(params)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ControlInterfaceOutput(tp.TypedDict):
    """
       ControlInterface model output spec.
    """
    output: SparkPayload

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ControlInterfaceConfig(InterfaceConfig):
    """
        Abstract ControlInterface model configuration class.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=ControlInterfaceConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ControlInterface(Interface, abc.ABC, tp.Generic[ConfigT]):
    """
        Abstract ControlInterface model.
    """
    config: ConfigT

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args: SparkPayload, **kwargs) -> ControlInterfaceOutput:
        """
            Control operation.
        """
        pass

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################