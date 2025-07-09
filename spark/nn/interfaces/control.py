#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax
import jax.numpy as jnp
from math import prod
from typing import TypedDict, Dict, List
from spark.core.specs import InputSpecs, OutputSpecs
from spark.core.variable_containers import Constant
from spark.core.shape import bShape, Shape, normalize_shape
from spark.core.registry import register_module
from spark.core.payloads import SparkPayload
from spark.nn.interfaces.base import Interface

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic OutputInterface output contract.
class ControlInterfaceOutput(TypedDict):
    output: SparkPayload

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ControlFlowInterface(Interface, abc.ABC):
    """
        Abstract control flow model.
    """

    def __init__(self, 
                 **kwargs):
        # Initialize super.
        super().__init__(**kwargs)

    # Override input specs
    def get_input_specs(self) -> Dict[str, InputSpecs]:
        """
            Returns a dictionary mapping logical input port names to their InputSpecs.
        """
        input_specs = {}
        for it, shape in enumerate(self._input_shapes):
            input_specs[f'stream_{it}'] = InputSpecs(
                payload_type=self._payload_type,
                shape=shape,
                is_optional=False,
                dtype=self._dtype,
                description=f'Input port for stream_{it}',
            )
        return input_specs

    # Override input specs
    def get_output_specs(self) -> Dict[str, OutputSpecs]:
        """
            Returns a dictionary mapping logical output port names to their OutputSpecs.
        """
        output_specs = {}
        for it, shape in enumerate(self._output_shapes):
            output_specs[f'stream_{it}'] = OutputSpecs(
                payload_type=self._payload_type,
                shape=shape,
                dtype=self._dtype,
                description=f'Output port for stream_{it}',
            )
        return output_specs

    @property
    def input_shapes(self,) -> List[Shape]:
        return self._input_shapes

    @property
    def output_shapes(self,) -> List[Shape]:
        return self._output_shapes

    @abc.abstractmethod
    def __call__(self, *args: SparkPayload, **kwargs) -> Dict[str, SparkPayload]:
        """
            Computes the control flow operation.
        """
        pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class Merger(ControlFlowInterface):
    """
        Combines several streams of inputs of the same type into a single stream.
    """

    def __init__(self, 
                 input_shapes: list[bShape],
                 **kwargs):
		# Initialize super.
        super().__init__(**kwargs)
        # Initialize shapes
        self._input_shapes = [normalize_shape(s) for s in input_shapes]
        self._output_shapes =[normalize_shape(sum([prod(x) for x in self._input_shapes]),)]
        # Main attributes
        self._payload_type = None

    def __call__(self, *streams: SparkPayload) -> ControlInterfaceOutput:
        """
            Computes the merge operation.
        """
        # Validate inputs (run only the when jit-compiled)
        if self._payload_type is None:
            self._payload_type = type(streams[0])
            if not issubclass(self._payload_type, SparkPayload):
                raise TypeError(f'"streams" must be of type "SparkPayload", got "{self._payload_type}".')
            for s in streams:
                if not isinstance(s, self._payload_type):
                    raise TypeError(f'stream must be of the same type, expected "{self._payload_type}" got "{type(s).__name__}".')
        # Control flow operation
        return {
            'stream': self._payload_type(jnp.concatenate([x.value.reshape(-1) for x in streams]))
        }
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class Sampler(ControlFlowInterface):
    """
        Sample a single input streams of inputs of the same type into a single stream.
        Indices are selected randomly and remain fixed.
    """

    def __init__(self, 
                 input_shape: bShape,
                 output_shape: bShape,
                 **kwargs):
        # Initialize super.
        super().__init__(**kwargs)
        self._input_shapes = normalize_shape(input_shape)
        self._output_shapes = [normalize_shape(output_shape)]
        # Build indices mask
        self._indices = Constant(jax.random.randint(self.get_rng_keys(1), self._output_shapes[0], 
                                                    minval=0, maxval=prod(self._input_shapes)), dtype=jnp.uint32)
        # Main attributes
        self._payload_type = None

    @property
    def indices(self,) -> jax.Array:
        return self._indices.value

    def __call__(self, stream: SparkPayload) -> ControlInterfaceOutput:
        """
            Computes the sample operation.
        """
        if self._payload_type is None:
            if not issubclass(self._payload_type, SparkPayload):
                raise TypeError(f'"stream" must be of type "SparkPayload", got "{type(stream).__name__}".')
            self._payload_type = type(stream)
        # Control flow operation
        return {
            'stream': self._payload_type(stream.value.reshape(-1)[self.indices])
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################