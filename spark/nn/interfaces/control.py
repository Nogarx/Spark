#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax
import jax.numpy as jnp
from math import prod
from typing import TypedDict, Dict, List
from spark.core.specs import InputSpec, OutputSpec
from spark.core.variable_containers import SparkConstant
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
    def get_input_specs(self) -> Dict[str, InputSpec]:
        """
            Returns a dictionary mapping logical input port names to their InputSpec.
        """
        input_specs = {}
        for it, shape in enumerate(self._input_shapes):
            input_specs[f'stream_{it}'] = InputSpec(
                payload_type=self._payload_type,
                shape=shape,
                is_optional=False,
                dtype=self._dtype,
                description=f'Input port for stream_{it}',
            )
        return input_specs

    # Override input specs
    def get_output_specs(self) -> Dict[str, OutputSpec]:
        """
            Returns a dictionary mapping logical output port names to their OutputSpec.
        """
        output_specs = {}
        for it, shape in enumerate(self._output_shapes):
            output_specs[f'stream_{it}'] = OutputSpec(
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
                 num_inputs: int,
                 payload_type: SparkPayload,
                 **kwargs):
		# Initialize super.
        super().__init__(**kwargs)
        # Intialize variables.
        self.num_inputs = num_inputs
        self.set_fallback_payload_type(payload_type)

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Validate payloads types.
        for key, value in input_specs.items():
            if self._default_payload_type != value.payload_type:
                raise TypeError(f'Expected "payload" to be of same type "{self._default_payload_type}" '
                                f'but input spec "{key}" is of type "{value.payload_type}".')

    def __call__(self, inputs: list[SparkPayload]) -> ControlInterfaceOutput:
        """
            Computes the merge operation.
        """
        # Control flow operation
        return {
            'output': self._default_payload_type(jnp.concatenate([x.value.reshape(-1) for x in inputs]))
        }
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class MergerReshape(ControlFlowInterface):
    """
        Combines several streams of inputs of the same type into a single stream.
    """

    def __init__(self, 
                 num_inputs: int,
                 reshape: Shape,
                 payload_type: SparkPayload,
                 **kwargs):
		# Initialize super.
        super().__init__(**kwargs)
        # Intialize variables.
        self.reshape = normalize_shape(reshape)
        self.num_inputs = num_inputs
        self.set_fallback_payload_type(payload_type)

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Validate payloads types.
        if self._default_payload_type != input_specs['inputs'].payload_type:
            raise TypeError(f'Expected "payload" to be of same type "{self._default_payload_type}" '
                            f'but input spec "inputs" is of type "{input_specs['inputs'].payload_type}".')
        # Validate can reshape.
        try:
            jnp.concatenate([jnp.zeros(s).reshape(-1) for s in input_specs['inputs'].shape]).reshape(self.reshape)
        except:
            raise ValueError(f'Shapes {input_specs['inputs'].shape} are not broadcastable to {self.reshape}')


    def __call__(self, inputs: list[SparkPayload]) -> ControlInterfaceOutput:
        """
            Computes the merge operation.
        """
        # Control flow operation
        return {
            'output': self._default_payload_type(jnp.concatenate([x.value.reshape(-1) for x in inputs]).reshape(self.reshape))
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class Sampler(ControlFlowInterface):
    """
        Sample a single input streams of inputs of the same type into a single stream.
        Indices are selected randomly and remain fixed.
    """

    def __init__(self, 
                 sample_size: int,
                 **kwargs):
        # Initialize super.
        super().__init__(**kwargs)
        # Initialize variables
        self.sample_size = sample_size

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Initialize shapes
        input_shape = normalize_shape(input_specs['input'].shape)
        # Initialize variables
        self._indices = SparkConstant(jax.random.randint(self.get_rng_keys(1), 
                                                         self.sample_size, 
                                                         minval=0, 
                                                         maxval=prod(input_shape)), 
                                                         dtype=jnp.uint32)

    @property
    def indices(self,) -> jax.Array:
        return self._indices.value

    def __call__(self, input: SparkPayload) -> ControlInterfaceOutput:
        """
            Computes the sample operation.
        """
        # Sample
        sample = type(input)(input.value.reshape(-1)[self.indices])
        return {
            'output': sample
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################