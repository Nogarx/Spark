#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import abc
import jax
import jax.numpy as jnp
from math import prod
from typing import Dict, List, TypedDict
from spark.nn.interfaces.base import Interface
from spark.core.tracers import SaturableTracer, SaturableDoubleTracer
from spark.core.payloads import SparkPayload, SpikeArray, FloatArray
from spark.core.variable_containers import Variable
from spark.core.shape import bShape, Shape, normalize_shape
from spark.core.registry import register_module
from spark.nn.interfaces.base import Interface

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic OutputInterface output contract.
class OutputInterfaceOutput(TypedDict):
    output: FloatArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class OutputInterface(Interface, abc.ABC):
    """
        Abstract output interface model.
    """

    def __init__(self, 
                 **kwargs):
        # Main attributes
        super().__init__(**kwargs)

    @abc.abstractmethod
    def __call__(self, *args: SpikeArray, **kwargs) -> Dict[str, SparkPayload]:
        """
            Transform incomming spikes into a output signal.
        """
        pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ExponentialIntegrator(OutputInterface):
    """
        Transforms a discrete spike signal to a continuous signal.
        This transformation assumes a very simple integration model model without any type of adaptation or plasticity.
        Spikes are grouped into k non-overlaping clusters and every neuron contributes the same amount to the ouput.
    """

    def __init__(self, 
                 input_shape: bShape,
                 output_dim: int,
                 saturation_freq: float = 50,   # [Hz]
                 tau: float = 20,               # [ms]
                 shuffle: bool = True,
                 smooth_trace: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        # Initialize shapes
        self._input_shape = normalize_shape(prod(normalize_shape(input_shape)))
        self._output_shape = normalize_shape(output_dim)
        # Initialize internal variables
        self._shuffle = shuffle
        self._tau = tau
        self._saturation_freq = saturation_freq
        self._smooth_trace = smooth_trace
        # Output mapping.
        in_dim, out_dim = self._input_shape[0], self._output_shape[0]
        base = in_dim // out_dim
        remainder = in_dim % out_dim
        counts = jnp.concatenate([jnp.full(remainder, base + 1, dtype=jnp.int32),
                                  jnp.full(out_dim - remainder, base, dtype=jnp.int32)])
        output_map = jnp.repeat(jnp.arange(out_dim, dtype=jnp.int32), counts)
        if not self._shuffle:
            output_map = output_map
        else: 
            output_map = jax.random.permutation(self.get_rng_keys(1), output_map)
        self._indices = Variable(output_map, dtype=jnp.uint32)
        # Initialize tracer
        if self._smooth_trace:
            self.I = SaturableDoubleTracer(shape=self._output_shape, 
                                           tau_1=self._tau, 
                                           tau_2=10,
                                           scale_1=(1000/self._saturation_freq)/(self._dt*self._tau*counts), 
                                           scale_2=1/10,
                                           dt=self._dt, 
                                           dtype=self._dtype)
        else:
            self.I = SaturableTracer(shape=self._output_shape, 
                                     tau=self._tau, 
                                     scale=(1000/self._saturation_freq)/(self._dt*self._tau*counts), 
                                     dt=self._dt, 
                                     dtype=self._dtype)

    @property
    def input_shapes(self,) -> List[Shape]:
        return [self._input_shape]

    @property
    def output_shapes(self,) -> List[Shape]:
        return [self._output_shape]

    def __call__(self, input_spikes: SpikeArray) -> OutputInterfaceOutput:
        # Flat array
        x = input_spikes.value.reshape(-1)
        # Count spikes in each output group.
        x = jax.ops.segment_sum(x, self._indices, 
                                indices_are_sorted=(not self._shuffle), 
                                num_segments=self._output_shape[0])
        # Integrate
        output = self.I(x)
        return {
            'output': FloatArray(output)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################