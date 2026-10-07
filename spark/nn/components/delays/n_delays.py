#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs
    
import jax
import numpy as np
import jax.numpy as jnp
import dataclasses as dc
import typing as tp
import spark.core.utils as utils
from math import prod, ceil
from spark.core.payloads import SpikeArray
from spark.core.backend import Variable, Constant
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.initializers.common import UniformInitializerConfig
from spark.nn.initializers.base import Initializer, InitializerConfig
from spark.nn.components.delays.base import Delays, DelaysOutput, DelaysConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class NDelaysConfig(DelaysConfig):
    """
        Configuration for `NDelays`.

        Parameters
        ----------
        max_delay : float, default 8.0
            Longest delay the buffer can hold, in ms. The buffer holds ``ceil(max_delay / dt)``
            past steps, which bounds every drawn delay.
        delays : jax.Array or Initializer, default UniformInitializerConfig()
            Delay of every presynaptic unit, in steps. Drawn over ``[1, ceil(max_delay / dt)]``
            when an initializer is given. A given array is taken as it is, and every delay in it
            lies in that range.
    """

    max_delay: float = dc.field(
        default = 8.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Maximum synaptic delay. Note: Final max delay is computed as ⌈max/dt⌉.',
        })
    delays: jax.Array | Initializer = dc.field(
        default_factory = UniformInitializerConfig,
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Synaptic delays array / initializer method.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

def delays_kernel(config: NDelaysConfig, key: jax.Array, shape: tuple[int, ...], longest: int) -> Constant:
    """
        Returns the delay of every entry of a kernel, in steps, within ``[1, longest]``.

        Parameters
        ----------
        config : NDelaysConfig
            Configuration holding the delays, or the initializer drawing them.
        key : jax.Array
            PRNG key of the draw.
        shape : tuple of int
            Shape of the kernel.
        longest : int
            Longest delay, ``ceil(max_delay / dt)`` steps.

        Returns
        -------
        Constant
            The kernel, in the smallest unsigned dtype holding ``longest``.

        Raises
        ------
        ValueError
            If a delay of a given array lies outside ``[1, longest]``.

        Notes
        -----
        A given array is taken as it is. An initializer draws the delay less one step, scaled to
        ``longest``, and the draw is clipped to the range: the default uniform draw covers
        ``[1, longest]`` evenly, and no initializer gives a delay of zero steps, or one the buffer
        does not hold.
    """
    dtype = np.min_scalar_type(longest)
    if isinstance(config.delays, (Initializer, InitializerConfig)):
        drawn = config.init.delays(key=key, shape=shape, dtype=jnp.int32, scale=longest, min_value=0)
        return Constant(jnp.clip(drawn + 1, 1, longest), dtype=dtype)
    delays = np.broadcast_to(np.asarray(config.init.delays()), shape)
    if delays.size and (delays.min() < 1 or delays.max() > longest):
        raise ValueError(
            f'Delays are of 1 to {longest} steps, ceil(max_delay / dt); the delays given are of {delays.min()} to '
            f'{delays.max()} steps.'
        )
    return Constant(delays, dtype=dtype)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class NDelays(Delays):
    """
        Conduction delay attached to the presynaptic unit.

        Every spike a unit emits reaches all of its targets after the same number of steps. If
        unit A fires, everything listening to A sees the spike ``k_A`` steps later, with ``k_A``
        the delay of A alone.

        Parameters
        ----------
        config : NDelaysConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        in_spikes : SpikeArray
            Spikes emitted on this step.

        Output Ports
        ------------
        out_spikes : SpikeArray
            Spikes due on this step, of the same shape as the input.

        Properties
        ----------
        kernel : IntegerArray
            Delay of every presynaptic unit, in steps rather than in ms. Writable.

        Notes
        -----
        Spikes are held in a ring buffer of ``ceil(max_delay / dt) + 1`` steps, the current one and
        every step a delay reaches back to, bit-packed eight units to a byte, so the buffer costs one
        bit per unit per step. Reading is a gather at the per-unit offset, which makes the cost
        independent of the delay values.

        See Also
        --------
        N2NDelays : One delay per (postsynaptic, presynaptic) pair.
    """
    config: NDelaysConfig

    def __init__(self, config: NDelaysConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, in_spikes: SpikeArray):
        # Initialize shapes
        self._shape = utils.validate_shape(in_spikes.shape)
        self._units = prod(self._shape)
        # Initialize varibles
        self.max_delay = self.config.max_delay
        self._buffer_size = int(ceil(self.max_delay / self._dt)) + 1
        num_bytes = (self._units + 7) // 8
        self._padding = (0, num_bytes * 8 - self._units)
        self._bitmask = Variable(jnp.zeros((self._buffer_size, num_bytes)), dtype=jnp.uint8)
        self._current_idx = Variable(0, dtype=jnp.int32)
        # Initialize kernel
        self._kernel = delays_kernel(self.config, self.get_rng_keys(1), (self._units,), self._buffer_size - 1)

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
        delay_idx = (self._current_idx.value - self._kernel.value - 1) % self._buffer_size
        selected_bytes = self._bitmask.value[delay_idx, byte_indices]
        selected_bits = (selected_bytes >> bit_indices) & 1
        return SpikeArray(
            selected_bits.reshape(self._shape), 
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
        """
            Stores the incoming spikes and returns the ones due on this step.

            Parameters
            ----------
            in_spikes : SpikeArray
                Spikes emitted on this step.

            Returns
            -------
            DelaysOutput
                Dictionary with one entry, ``out_spikes``, the spikes whose delay elapsed on this
                step.
        """
        self._push(in_spikes)
        out_spikes = self._gather(in_spikes.inhibition_mask)
        return {
            'out_spikes': out_spikes
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################