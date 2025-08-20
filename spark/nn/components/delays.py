#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import abc
import jax
import jax.numpy as jnp
from typing import TypedDict
from math import prod
from spark.core.payloads import SpikeArray
from spark.core.variables import Variable, Constant
from spark.core.shape import Shape, normalize_shape
from spark.core.registry import register_module
from spark.nn.components.base import Component
from spark.nn.initializers.base import Initializer
from spark.nn.initializers.delay import uniform_delay_initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic Delays output contract.
class DelaysOutput(TypedDict):
    out_spikes: SpikeArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Delays(Component):
    """
        Abstract synaptic delay model.
    """

    def __init__(self, 
                 **kwargs):
        # Initialize super.
        super().__init__(**kwargs)
    
    @abc.abstractmethod
    def _push(self, spikes: SpikeArray) -> None:
        """
            Push operation.
        """
        pass

    @abc.abstractmethod
    def _gather(self,) -> SpikeArray:
        """
            Gather operation.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
            Resets component state.
        """
        pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class DummyDelays(Delays):
    """
        A dummy delay that is equivalent to no delay at all, it simply forwards its input.
        This is a convinience module, that should be avoided whenever is possible and its 
        only purpose is to simplify some scenarios.

        Init:

        Input:
            in_spikes: SpikeArray
            
        Output:
            out_spikes: SpikeArray
    """

    def __init__(self, 
                 **kwargs):
        # Initialize super.
        super().__init__(**kwargs)

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize shapes
        self._shape = normalize_shape(input_specs['in_spikes'].shape)
        self._units = prod(self._shape)
        # Initialize varibles
        self.spikes = Variable(jnp.zeros(self._shape, self._dtype), dtype=self._dtype)

    def reset(self) -> None:
        """
            Resets component state.
        """
        self.spikes.value = jnp.zeros(self._shape, self._dtype)

    def _push(self, in_spikes: SpikeArray) -> None:
        """
            Push operation.
        """
        self.spikes.value = in_spikes.value

    def _gather(self,) -> SpikeArray:
        """
            Gather operation.
        """
        return SpikeArray(self.spikes.value)

    # Override call, no need to store anything. Push and Gather are defined just for the sake of completion.
    def __call__(self, in_spikes: SpikeArray) -> DelaysOutput:
        return {
            'out_spikes': in_spikes,
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class NDelays(Delays):
    """
        Data structure for spike storage and retrival for efficient neuron spike delay implementation.
        This synaptic delay model implements a generic conduction delay of the outputs spikes of neruons. 
        Example: Neuron A fires, every neuron that listens to A recieves its spikes K timesteps later,
                 neuron B fires, every neuron that listens to B recieves its spikes L timesteps later.

        Init:
            max_delay: int
            delay_kernel_initializer: Initializer

        Input:
            in_spikes: SpikeArray
            
        Output:
            out_spikes: SpikeArray
    """

    def __init__(self, 
                 max_delay: int, 
                 delay_kernel_initializer: Initializer = uniform_delay_initializer, 
                 **kwargs):
        # Initialize super.
        super().__init__(**kwargs)
        # Sanity checks
        if not isinstance(max_delay, int) and max_delay > 0:
            raise ValueError(f'"max_delay" must be a positive integer, got {max_delay}.')
        # Initialize varibles
        self.max_delay = max_delay
        self.delay_kernel_initializer = delay_kernel_initializer

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize shapes
        self._shape = normalize_shape(input_specs['in_spikes'].shape)
        self._units = prod(self._shape)
        # Initialize varibles
        self._buffer_size = self.max_delay
        num_bytes = (self._units + 7) // 8
        self._padding = (0, num_bytes * 8 - self._units)
        self._bitmask = Variable(jnp.zeros((self._buffer_size, num_bytes)), dtype=jnp.uint8)
        self._current_idx = Variable(0, dtype=jnp.int32)
        # Initialize delay kernel
        delay_kernel = self.delay_kernel_initializer(scale=self._buffer_size)(self.get_rng_keys(1), self._units)
        self.delay_kernel = Constant(delay_kernel, dtype=jnp.uint8)

    def reset(self) -> None:
        """
            Resets component state.
        """
        self._bitmask.value = jnp.zeros_like(self._bitmask.value, dtype=jnp.uint8)
        self._current_idx.value = jnp.zeros_like(self._current_idx.value, dtype=jnp.int32)

    def _push(self, spikes: SpikeArray) -> None:
        """
            Push operation.
        """
        # Pad and pack the binary vector (MSB-first)
        #padded_vec = jnp.pad(spikes.value.reshape(-1), self._padding)
        padded_vec = jnp.pad(jnp.abs(spikes.value).reshape(-1), self._padding)
        reshaped = padded_vec.reshape(-1, 8)
        bits = jnp.left_shift(1, jnp.arange(7, -1, step=-1, dtype=jnp.uint8))
        new_bitmask_row = jnp.dot(reshaped.astype(jnp.uint8), bits).astype(jnp.uint8)
        # Update the buffer
        self._bitmask.value = self._bitmask.value.at[self._current_idx.value].set(new_bitmask_row)
        self._current_idx.value = (self._current_idx.value + 1) % self._buffer_size

    def _gather(self, sign: jax.Array) -> SpikeArray:
        """
            Gather operation.
        """
        j_indices = jnp.arange(self._units)
        byte_indices = j_indices // 8
        bit_indices = 7 - (j_indices % 8)  # MSB-first adjustment
        delay_idx = (self._current_idx.value - self.delay_kernel.value - 1) % self._buffer_size
        selected_bytes = self._bitmask.value[delay_idx, byte_indices]
        selected_bits = (selected_bytes >> bit_indices) & 1
        return SpikeArray(selected_bits.astype(self._dtype).reshape(self._shape) * sign)

    def get_dense(self,) -> jax.Array:
        """
            Convert bitmask to dense vector (aligned with MSB-first packing).
        """
        # Unpack all bitmasks into bits (shape: [buffer_size, num_bytes, 8])
        unpacked = jnp.unpackbits(self._bitmask.value, axis=1, count=self._units)
        # Flatten to [buffer_size, num_bytes*8] and truncate to vector_size
        return unpacked.reshape(self._buffer_size, -1)[:, :self._units].astype(self._dtype).reshape((self._buffer_size,self._units))

    def __call__(self, in_spikes: SpikeArray) -> DelaysOutput:
        self._push(in_spikes)
        out_spikes = self._gather(1 - 2 *jnp.signbit(in_spikes.value))
        return {
            'out_spikes': out_spikes
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class N2NDelays(Delays):
    """
        Data structure for spike storage and retrival for efficient neuron to neuron spike delay implementation.
        This synaptic delay model implements specific conduction delays between specific neruons. 
        Example: Neuron A fires and neuron B, C, and D listens to A; neuron B recieves A's spikes I timesteps later,
                 neuron C recieves A's spikes J timesteps later and neuron D recieves A's spikes K timesteps later.

        Init:
            max_delay: int
            output_shape: Shape
            delay_kernel_initializer: Initializer

        Input:
            in_spikes: SpikeArray
            
        Output:
            out_spikes: SpikeArray
    """

    def __init__(self, 
                 max_delay: int, 
                 output_shape: Shape, 
                 delay_kernel_initializer: Initializer = uniform_delay_initializer, 
                 **kwargs):
        # Sanity checks
        if not isinstance(max_delay, int) and max_delay > 0:
            raise ValueError(f'"max_delay" must be a positive integer, got {max_delay}.')
        # Initialize super.
        super().__init__(**kwargs)
        # Initialize varibles
        self.max_delay = max_delay
        self.output_shape = normalize_shape(output_shape)
        self.delay_kernel_initializer = delay_kernel_initializer

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize shapes
        self._in_shape = normalize_shape(input_specs['in_spikes'].shape)
        self._kernel_shape = normalize_shape((prod(self.output_shape), prod(self._in_shape)))
        self._units = prod(self._in_shape)
        # Initialize varibles
        self._buffer_size = self.max_delay
        num_bytes = (self._units + 7) // 8
        self._padding = (0, num_bytes * 8 - self._units)
        self._bitmask = Variable(jnp.zeros((self._buffer_size, num_bytes)), dtype=jnp.uint8)
        self._current_idx = Variable(0, dtype=jnp.int32)
        # Initialize kernel
        delay_kernel = self.delay_kernel_initializer(scale=self._buffer_size)(self.get_rng_keys(1), self._kernel_shape)
        self.delay_kernel = Constant(delay_kernel, dtype=jnp.uint8)

    def reset(self) -> None:
        """
            Resets component state.
        """
        self._bitmask.value = jnp.zeros_like(self._bitmask.value, dtype=jnp.uint8)
        self._current_idx.value = jnp.zeros_like(self._current_idx.value, dtype=jnp.int32)

    def _push(self, spikes: SpikeArray) -> None:
        """
            Push operation.
        """
        # Pad and pack the binary vector (MSB-first)
        #padded_vec = jnp.pad(spikes.value.reshape(-1), self._padding)
        padded_vec = jnp.pad(jnp.abs(spikes.value).reshape(-1), self._padding)
        reshaped = padded_vec.reshape(-1, 8)
        bits = jnp.left_shift(1, jnp.arange(7, -1, step=-1, dtype=jnp.uint8))
        new_bitmask_row = jnp.dot(reshaped.astype(jnp.uint8), bits).astype(jnp.uint8)
        # Update the buffer
        self._bitmask.value = self._bitmask.value.at[self._current_idx.value].set(new_bitmask_row)
        self._current_idx.value = (self._current_idx.value + 1) % self._buffer_size

    def _gather(self, sign: jax.Array) -> SpikeArray:
        """
            Gather operation.
        """
        j_indices = jnp.arange(self._units)
        byte_indices = j_indices // 8
        bit_indices = 7 - (j_indices % 8)  # MSB-first adjustment
        delay_idx = (self._current_idx.value - self.delay_kernel.value - 1) % self._buffer_size
        selected_bytes = self._bitmask.value[delay_idx, byte_indices]
        selected_bits = (selected_bytes >> bit_indices) & 1
        return SpikeArray(selected_bits.astype(self._dtype).reshape(self.output_shape+self._in_shape) * sign)

    def get_dense(self,) -> jax.Array:
        """
            Convert bitmask to dense vector (aligned with MSB-first packing).
        """
        # Unpack all bitmasks into bits (shape: [buffer_size, num_bytes, 8])
        unpacked = jnp.unpackbits(self._bitmask.value, axis=1, count=self._units)
        # Flatten to [buffer_size, num_bytes*8] and truncate to vector_size
        return unpacked.reshape(self._buffer_size, -1)[:, :self._units].astype(self._dtype).reshape((self._buffer_size,self._units))

    def __call__(self, in_spikes: SpikeArray) -> DelaysOutput:
        self._push(in_spikes)
        out_spikes = self._gather(1 - 2 *jnp.signbit(in_spikes.value))
        return {
            'out_spikes': out_spikes
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################