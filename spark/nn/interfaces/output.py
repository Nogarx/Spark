#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.payloads import SparkPayload
    from spark.core.specs import InputSpec

import abc
import jax
import jax.numpy as jnp
from math import prod
from typing import Dict, List, TypedDict
from spark.nn.interfaces.base import Interface
from spark.core.tracers import SaturableTracer, SaturableDoubleTracer
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.variable_containers import SparkVariable
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
                 num_outputs: int,
                 saturation_freq: float = 50,   # [Hz]
                 tau: float = 20,               # [ms]
                 shuffle: bool = True,
                 smooth_trace: bool = True,
                 **kwargs):
        # Initialize super.
        super().__init__(**kwargs)
        # Initialize internal variables
        self.num_outputs = num_outputs
        self.saturation_freq = saturation_freq
        self.tau = tau
        self.shuffle = shuffle
        self.smooth_trace = smooth_trace

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Output mapping.
        in_dim = prod(input_specs['spikes'].shape)
        out_dim = self.num_outputs
        base = in_dim // out_dim
        remainder = in_dim % out_dim
        counts = jnp.concatenate([jnp.full(remainder, base + 1, dtype=jnp.int32),
                                  jnp.full(out_dim - remainder, base, dtype=jnp.int32)])
        output_map = jnp.repeat(jnp.arange(out_dim, dtype=jnp.int32), counts)
        if not self.shuffle:
            output_map = output_map
        else: 
            output_map = jax.random.permutation(self.get_rng_keys(1), output_map)
        self._indices = SparkVariable(output_map, dtype=jnp.uint32)

        # Initialize tracer
        if self.smooth_trace:
            self.trace = SaturableDoubleTracer(shape=self.num_outputs, 
                                               tau_1=self.tau, 
                                               tau_2=10,
                                               scale_1=(1000/self.saturation_freq)/(self._dt*self.tau*counts), 
                                               scale_2=1/10,
                                               dt=self._dt, 
                                               dtype=self._dtype)
        else:
            self.trace = SaturableTracer(shape=self.num_outputs, 
                                         tau=self.tau, 
                                         scale=(1000/self.saturation_freq)/(self._dt*self.tau*counts), 
                                         dt=self._dt, 
                                         dtype=self._dtype)
            
    def __call__(self, spikes: SpikeArray) -> OutputInterfaceOutput:
        # Flat array
        x = spikes.value.reshape(-1)
        # Count spikes in each output group.
        x = jax.ops.segment_sum(x, self._indices, 
                                indices_are_sorted=(not self.shuffle), 
                                num_segments=self.num_outputs)
        # Integrate
        output = self.trace(x)
        return {
            'output': FloatArray(output)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################