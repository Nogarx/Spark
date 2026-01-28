#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs

import jax
import jax.numpy as jnp
import dataclasses as dc
import typing as tp
import spark.core.utils as utils
from math import prod, ceil
from spark.core.payloads import SpikeArray
from spark.core.variables import Variable, Constant
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator
from spark.nn.components.delays.base import Delays, DelaysOutput
from spark.nn.components.delays.n_delays import NDelaysConfig
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class N2NDelaysConfig(NDelaysConfig):
    """
       N2NDelays configuration class.
    """

    units: tuple[int, ...] = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Shape of the postsynaptic pool of neurons.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class N2NDelays(Delays):
    """
        Data structure for spike storage and retrival for efficient neuron to neuron spike delay implementation.
        This synaptic delay model implements specific conduction delays between specific neruons. 
        Example: Neuron A fires and neuron B, C, and D listens to A; neuron B recieves A's spikes I timesteps later,
                 neuron C recieves A's spikes J timesteps later and neuron D recieves A's spikes K timesteps later.

        Init:
            units: tuple[int, ...]
            max_delay: float
            delays: jnp.ndarray | Initializer

        Input:
            in_spikes: SpikeArray
            
        Output:
            out_spikes: SpikeArray
    """
    config: N2NDelaysConfig

    def __init__(self, config: N2NDelaysConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, input_specs: dict[str, PortSpecs]):
        # Initialize shapes
        self._in_shape = utils.validate_shape(input_specs['in_spikes'].shape)
        self.output_shape = utils.validate_shape(self.config.units)
        self._kernel_shape = utils.validate_shape((prod(self.output_shape), prod(self._in_shape)))
        self._units = prod(self._in_shape)
        # Initialize varibles
        self.max_delay = self.config.max_delay
        self._buffer_size = int(ceil(self.max_delay / self._dt))
        num_bytes = (self._units + 7) // 8
        self._padding = (0, num_bytes * 8 - self._units)
        self._bitmask = Variable(jnp.zeros((self._buffer_size, num_bytes)), dtype=jnp.uint8)
        self._current_idx = Variable(0, dtype=jnp.int32)
        # Initialize kernel
        delays_kernel = self.config.delays.init(
            init_kwargs = {'scale':self._buffer_size+1, 'min_value':1,},
            key=self.get_rng_keys(1), shape=self._kernel_shape, dtype=jnp.uint8,
        )
        self.delays_kernel = Constant(delays_kernel, dtype=jnp.uint8)

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
        padded_vec = jnp.pad(spikes.spikes.reshape(-1), self._padding)
        reshaped = padded_vec.reshape(-1, 8)
        bits = jnp.left_shift(1, jnp.arange(7, -1, step=-1, dtype=jnp.uint8))
        new_bitmask_row = jnp.dot(reshaped.astype(jnp.uint8), bits).astype(jnp.uint8)
        # Update the buffer
        self._bitmask.value = self._bitmask.value.at[self._current_idx.value].set(new_bitmask_row)
        self._current_idx.value = (self._current_idx.value + 1) % self._buffer_size

    def _gather(self, inhibition_mask: jax.Array) -> SpikeArray:
        """
            Gather operation.
        """
        j_indices = jnp.arange(self._units)
        byte_indices = j_indices // 8
        bit_indices = 7 - (j_indices % 8)  # MSB-first adjustment
        delay_idx = (self._current_idx.value - self.delays_kernel.value - 1) % self._buffer_size
        selected_bytes = self._bitmask.value[delay_idx, byte_indices]
        selected_bits = (selected_bytes >> bit_indices) & 1
        return SpikeArray(
            selected_bits.reshape(self.output_shape+self._in_shape), 
            inhibition_mask=inhibition_mask, 
            async_spikes=True,
        )

    def get_dense(self,) -> jax.Array:
        """
            Convert bitmask to dense vector (aligned with MSB-first packing).
        """
        # Unpack all bitmasks into bits (shape: [buffer_size, num_bytes, 8])
        unpacked = jnp.unpackbits(self._bitmask.value, axis=1, count=self._units)
        # Flatten to [buffer_size, num_bytes*8] and truncate to vector_size
        return unpacked.reshape(self._buffer_size, -1)[:, :self._units].reshape((self._buffer_size,self._units))

    def __call__(self, in_spikes: SpikeArray) -> DelaysOutput:
        self._push(in_spikes)
        out_spikes = self._gather(in_spikes.inhibition_mask)
        return {
            'out_spikes': out_spikes
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################