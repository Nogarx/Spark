#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import typing as tp
import dataclasses as dc
import spark.core.utils as utils
from math import prod
from spark.core.module import SparkModule
from spark.core.payloads import SpikeArray
from spark.core.config import DefaultSparkConfig
from spark.core.config_validation import TypeValidator

# NOTE: Although the Neuron Controller is the ideal way to build new neuronal models, we suspect that
# sometimes it may be necessary to fallback a proper SparkModule, this code serves this purpose.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class NeuronOutput(tp.TypedDict):
    """
        Output ports of a neuron model.

        Attributes
        ----------
        out_spikes : SpikeArray
            Spikes emitted by the pool on this step.
    """
    out_spikes: SpikeArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class NeuronModuleConfig(DefaultSparkConfig):
    """
        Base configuration for neuron models written as a single module.

        Parameters
        ----------
        units : tuple of int
            Shape of the pool of neurons.
        seed : int, optional
            Seed for internal random draws. Drawn from the operating system when omitted.
        dtype : DTypeLike, default jnp.float16
            Dtype used for the internal state.
        dt : float, default 1.0
            Integration step, in ms.
    """
    units: tuple[int, ...] = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Shape of the pool of neurons.',
        })
    
    # TODO: Manual override to synchronize all time integration constants across the controller.
    # This solution is probably good enough but it is not clear that will not clash with other user intentions.
    # A similar situation is present in Neuron.__post_init__
    def __post_init__(self,) -> None:
        pass
        # Synchronize dt's. NOTE: Skip validation, otherwise will fall into an infinite loop.
        #self = self.merge(_s_dt=self.dt, _s_units=self.units)

ConfigT = tp.TypeVar("ConfigT", bound=NeuronModuleConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class NeuronModule(SparkModule, abc.ABC, tp.Generic[ConfigT]):
    """
        Base class for neuron models written as a single module.

        A neuron assembled in Python rather than declared as a wiring of components. The
        components are held as plain attributes and stepped by `__call__`, which is the fallback
        for a model that a `Neuron` controller cannot express.

        Parameters
        ----------
        config : NeuronModuleConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        in_spikes : SpikeArray
            Spikes arriving at the pool.

        Output Ports
        ------------
        out_spikes : SpikeArray
            Spikes emitted by the pool on this step.

        Notes
        -----
        Use this neuron model only when looking for specific information flows that may not be
        implemented with the default neuron controller.

        See Also
        --------
        Neuron : Controller-based neuron, declared as a wiring of components.
    """
    config: ConfigT

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)
        # Initialize shapes
        self.units = utils.validate_shape(self.config.units)
        self._units = prod(self.units)
        self._component_names: list[SparkModule] | None = None

    def _build(self, *abc_args, **kwargs):
        super()._build(*abc_args, **kwargs)
        self._build_component_list()
    
    def _build_component_list(self):
        """
            Inspect the object to collect the names of all child of type Component.
        """
        from spark.nn.components.base import Component

        self._component_names = []
        # Get all attribute names
        all_attr_names = []
        if hasattr(self, '__dict__'):
            all_attr_names = list(vars(self).keys())
        # Add attributes from __slots__ if they exist
        if hasattr(self, '__slots__'):
            all_attr_names.extend(self.__slots__)
        # Check the attribute's type
        for name in set(all_attr_names):
            try:
                if isinstance(getattr(self, name), Component):
                    self._component_names.append(name)
            except AttributeError:
                continue

    def reset(self):
        """
            Resets neuron states to their initial values.
        """
        # Build components list. 
        if self._component_names is None:
            self._build_component_list()
        # Reset components.
        for name in self._component_names:
            getattr(self, name).reset()

    @abc.abstractmethod
    def __call__(self, in_spikes: SpikeArray) -> NeuronOutput:
        """
            Advances the neuron one step.

            Parameters
            ----------
            in_spikes : SpikeArray
                Spikes arriving at the pool.

            Returns
            -------
            NeuronOutput
                Dictionary with one entry, ``out_spikes``, the spikes emitted by the pool.
        """
        pass

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################