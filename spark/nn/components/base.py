#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import typing as tp
from spark.core.module import SparkModule
from spark.core.config import DefaultSparkConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ComponentConfig(DefaultSparkConfig):
    """
        Base configuration for components.

        Parameters
        ----------
        seed : int, optional
            Seed for internal random draws. Drawn from the operating system when omitted.
        dtype : DTypeLike, default jnp.float16
            Dtype used for the internal state.
        dt : float, default 1.0
            Integration step, in ms. Overwritten by the enclosing controller.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

ConfigT = tp.TypeVar("ConfigT", bound=ComponentConfig)

class Component(SparkModule, abc.ABC, tp.Generic[ConfigT]):
    """
        Base class for the components a neuron is built from.

        Parameters
        ----------
        config : ComponentConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        **inputs : SparkPayload
            Declared by the concrete component through the signature of its ``__call__``.

        Output Ports
        ------------
        **outputs : SparkPayload
            Declared by the concrete component through the TypedDict its ``__call__`` returns.

        See Also
        --------
        Soma : Membrane potential and spike generation.
        Synapses : Presynaptic spikes to postsynaptic current.
        Delays : Conduction delays.
        Plasticity : Weight updates.
    """
    config: ConfigT

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################