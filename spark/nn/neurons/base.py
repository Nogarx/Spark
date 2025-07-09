#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
from typing import TypedDict, List
from spark.core.module import SparkModule
from spark.core.payloads import SpikeArray
from math import prod
from spark.core.shape import bShape, Shape, normalize_shape

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic Soma output contract.
class NeuronOutput(TypedDict):
    spikes: SpikeArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#


class Neuron(SparkModule):
    """
        Abstract neuronal model.
        This is more a convenience class used to synchronize data more easily.
    """

    def __init__(self, 
                 units: bShape,
                 input_shapes: List[bShape] = None,
                 **kwargs):
        super().__init__(**kwargs)
        # Initialize shapes
        self._input_shapes = [normalize_shape(s) for s in input_shapes] \
                if isinstance(input_shapes, list) else [normalize_shape(input_shapes)]
        self._output_shape = normalize_shape(units)
        self.units = prod(self._output_shape)

    @property
    def input_shapes(self,) -> List[Shape]:
        return self._input_shapes

    @property
    def output_shapes(self,) -> List[Shape]:
        return [self._output_shape]

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
        # Build components list. TODO: This should be called post init. 
        if self._component_names is not None:
            self._build_component_list()
        # Reset components.
        for name in self._component_names:
            getattr(self, name).reset()

    @abc.abstractmethod
    def __call__(self, *spikes: SpikeArray) -> NeuronOutput:
        """
            Update neuron's states and compute spikes.
        """
        pass

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################