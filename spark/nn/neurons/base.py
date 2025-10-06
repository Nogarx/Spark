#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import typing as tp
import dataclasses as dc
from math import prod
from spark.core.module import SparkModule
from spark.core.shape import Shape
from spark.core.payloads import SpikeArray
from spark.core.config import SparkConfig
from spark.core.config_validation import TypeValidator

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic Soma output contract.
class NeuronOutput(tp.TypedDict):
    out_spikes: SpikeArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class NeuronConfig(SparkConfig):
    units: Shape = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Shape of the pool of neurons.',
        })
ConfigT = tp.TypeVar("ConfigT", bound=NeuronConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Neuron(SparkModule, abc.ABC, tp.Generic[ConfigT]):
    """
        Abstract neuronal model.

        This is a convenience class used to synchronize data more easily.
        Can be thought as the equivalent of Sequential in standard ML frameworks.
    """
    config: ConfigT

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)
        # Initialize shapes
        self.units = Shape(self.config.units)
        self._units = prod(self.units)
        self._component_names: list[SparkModule] | None = None

    # TODO: This should be called post build. 
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
        pass

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################