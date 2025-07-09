#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax
import jax.numpy as jnp
from typing import TypedDict, Dict, List
from spark.core.payloads import SparkPayload, SpikeArray, FloatArray
from spark.core.variable_containers import Variable, Constant
from spark.core.shape import bShape, Shape, normalize_shape
from spark.core.registry import register_module
from spark.nn.interfaces.base import Interface

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic InputInterface output contract.
class InputInterfaceOutput(TypedDict):
    spikes: SpikeArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InputInterface(Interface, abc.ABC):
    """
        Abstract input interface model.
    """

    def __init__(self, 
                 **kwargs):
        # Main attributes
        super().__init__(**kwargs)

    @abc.abstractmethod
    def __call__(self, *args: SparkPayload, **kwargs) -> Dict[str, SparkPayload]:
        """
            Transform the input signal into an Spike signal.
        """
        pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class PoissonSpiker(InputInterface):
    """
        Transforms a continuous signal to a spiking signal.
        This transformation assumes a very simple poisson neuron model without any type of adaptation or plasticity.
    """

    def __init__(self, 
                 input_shape: bShape,
                 max_freq: float = 50, 	# [Hz]
                 **kwargs):
        super().__init__(**kwargs)
        # Initialize shapes
        self._shape = normalize_shape(input_shape)
        # Initialize variables
        self._scale = self._dt * (max_freq / 1000)

    @property
    def input_shapes(self,) -> List[Shape]:
        return [self._shape]

    @property
    def output_shapes(self,) -> List[Shape]:
        return [self._shape]

    def __call__(self, drive: FloatArray) -> InputInterfaceOutput:
        """
            Input interface operation.

            Input: A FloatArray of values in the range [0,1].
            Output: A SpikeArray of the same shape as the input.
        """
        spikes = (jax.random.uniform(self.get_rng_keys(1), shape=self._shape) < self._scale * drive.value).astype(self._dtype)
        return {
            'spikes': SpikeArray(spikes)
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class LinearSpiker(InputInterface):
    """
        Transforms a continuous signal to a spiking signal.
        This transformation assumes a very simple linear neuron model without any type of adaptation or plasticity.
        Units have a fixed refractory period and at maximum input signal will fire up to some fixed frequency.
    """

    def __init__(self, 
                 input_shape: bShape,
                 tau: float = 100, 	 	# [ms]
                 cd: float = 2.0,		# [ms]
                 max_freq: float = 50, 	# [Hz]
                 **kwargs):
        super().__init__(**kwargs)
        # Sanity checks
        if not isinstance(cd, float) and cd > 0:
            raise ValueError(f'"cd" must be a positive float, got {cd}.')
        # Initialize shapes
        self._shape = normalize_shape(input_shape)
        # Initialize variables
        exp_term = jnp.exp((1/tau) * ((1000-cd*max_freq) / max_freq)) # dt cancels out
        scale = ((1 / (exp_term - 1)) + 1)
        self._scale = Constant(scale, dtype=self._dtype)
        self._tau = Constant(tau, dtype=self._dtype)
        self._decay = Constant(jnp.exp(-self._dt / self._tau), dtype=self._dtype)
        self._gain = Constant(1 - self._decay, dtype=self._dtype)
        self._cooldown = Constant(cd * jnp.ones(shape=self._shape), dtype=self._dtype)
        self._refractory = Variable(self._cooldown, dtype=self._dtype)
        self.potential = Variable(jnp.zeros(shape=self._shape), dtype=self._dtype)

    @property
    def input_shapes(self,) -> List[Shape]:
        return [self._shape]

    @property
    def output_shapes(self,) -> List[Shape]:
        return [self._shape]

    def __call__(self, drive: FloatArray) -> InputInterfaceOutput:
        """
            Input interface operation.

            Input: A FloatArray of values in the range [0,1].
            Output: A SpikeArray of the same shape as the input.
        """
        # Update potential
        is_ready = jnp.greater_equal(self._refractory.value, self._cooldown).astype(self._dtype)
        dV = is_ready * self._tau * self._gain * self._scale * drive.value
        self.potential.value = self._decay * self.potential.value + dV
        # Spike
        spikes = (self.potential.value > self._tau).astype(self._dtype)
        # Reset neuron 
        self.potential.value = (1 - spikes) * self.potential.value
        # Set neuron refractory period.
        self._refractory.value = (1 - spikes) * (self._refractory.value + self._dt)
        return {
            'spikes': SpikeArray(spikes)
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class TopologicalPoissonSpiker(InputInterface):
    """
        Transforms a continuous signal to a spiking signal.
        This transformation maps a vector (a point in a hypercube) into a simple manifold with/without its borders glued.
        This transformation assumes a very simple poisson neuron model without any type of adaptation or plasticity.
    """

    def __init__(self, 
                 input_shape: bShape,
                 glue: jax.Array = jnp.array(0),
                 mins: jax.Array = jnp.array(0),
                 maxs: jax.Array = jnp.array(1),
                 resolution: int = 64,              # Num units
                 max_freq: float = 50, 	            # [Hz]
                 sigma: float = 1/32,
                 **kwargs):
        super().__init__(**kwargs)
        # Initialize shapes
        self._input_shape = normalize_shape(input_shape)
        self._output_shape = normalize_shape(self._input_shape + (resolution,))
        # Initialize variables
        self._scale = self._dt * (max_freq / 1000)
        self._glue = Constant(glue, dtype=jnp.bool_)
        self._mins = Constant(mins, dtype=self._dtype)
        self._maxs = Constant(maxs, dtype=self._dtype)
        self._sigma = Constant(sigma, dtype=self._dtype)
        self._space = Constant(jnp.linspace(jnp.zeros(self._input_shape), 
                                            jnp.pi*jnp.ones(self._input_shape), resolution), dtype=self._dtype)

    @property
    def input_shapes(self,) -> List[Shape]:
        return [self._input_shape]

    @property
    def output_shapes(self,) -> List[Shape]:
        return [self._output_shape]

    def __call__(self, drive: FloatArray) -> InputInterfaceOutput:
        """
            Input interface operation.

            Input: A FloatArray of values in the range [mins, maxs].
            Output: A SpikeArray of the same shape as the input.
        """
        # Transform input to [0, 1]
        x = (drive.value - self._mins) / (self._maxs - self._mins)
        x = jnp.where(self._glue.value, jnp.sin(self._space + x*jnp.pi), jnp.tanh(self._space - x*jnp.pi))
        x = jnp.exp( -(0.5 / self._sigma) * (x**2) ).T
        # Poisson process
        spikes = (jax.random.uniform(self.get_rng_keys(1), shape=self._output_shape) < self._scale * x).astype(self._dtype)
        return {
            'spikes': SpikeArray(spikes)
        }
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class TopologicalLinearSpiker(InputInterface):
    """
        Transforms a continuous signal to a spiking signal.
        This transformation maps a vector (a point in a hypercube) into a simple manifold with/without its borders glued.
        This transformation assumes a very simple linear neuron model without any type of adaptation or plasticity.
    """

    def __init__(self, 
                 input_shape: bShape,
                 glue: jax.Array = jnp.array(0),
                 mins: jax.Array = jnp.array(0),
                 maxs: jax.Array = jnp.array(1),
                 resolution: int = 64,              # Num units
                 tau: float = 100, 	 	            # [ms]
                 cd: float = 2.0,		            # [ms]
                 max_freq: float = 50, 	            # [Hz]
                 sigma: float = 1/32,
                 **kwargs):
        # Initialize super.
        super().__init__(**kwargs)
        # Initialize shapes
        self._input_shape = normalize_shape(input_shape)
        self._output_shape = normalize_shape(self._input_shape + (resolution,))
        # Initialize variables
        self._glue = Constant(glue, dtype=jnp.bool_)
        self._mins = Constant(mins, dtype=self._dtype)
        self._maxs = Constant(maxs, dtype=self._dtype)
        self._sigma = Constant(sigma, dtype=self._dtype)
        self._space = Constant(jnp.linspace(jnp.zeros(self._input_shape), 
                                            jnp.pi*jnp.ones(self._input_shape), resolution), dtype=self._dtype)
        exp_term = jnp.exp((1/tau) * ((1000-cd*max_freq) / max_freq)) # dt cancels out
        scale = ((1 / (exp_term - 1)) + 1)
        self._scale = Constant(scale, dtype=self._dtype)
        self._tau = Constant(tau, dtype=self._dtype)
        self._decay = Constant(jnp.exp(-self._dt / self._tau), dtype=self._dtype)
        self._gain = Constant(1 - self._decay, dtype=self._dtype)
        self._cooldown = Constant(cd * jnp.ones(shape=self._output_shape), dtype=self._dtype)
        self._refractory = Variable(self._cooldown, dtype=self._dtype)
        self.potential = Variable(jnp.zeros(shape=self._output_shape), dtype=self._dtype)

    @property
    def input_shapes(self,) -> List[Shape]:
        return [self._input_shape]

    @property
    def output_shapes(self,) -> List[Shape]:
        return [self._output_shape]

    def __call__(self, drive: FloatArray) -> InputInterfaceOutput:
        """
            Input interface operation.

            Input: A FloatArray of values in the range [mins, maxs].
            Output: A SpikeArray of the same shape as the input.
        """
        # Transform input to [0, 1]
        x = (drive.value - self._mins) / (self._maxs - self._mins)
        x = jnp.where(self._glue.value, jnp.sin(self._space + x*jnp.pi), jnp.tanh(self._space - x*jnp.pi))
        x = jnp.exp( -(0.5 / self._sigma) * (x**2) ).T
        # Update potential
        is_ready = jnp.greater_equal(self._refractory.value, self._cooldown).astype(self._dtype)
        dV = is_ready * self._tau * self._gain * self._scale * x
        self.potential.value = self._decay * self.potential.value + dV
        # Spike
        spikes = jnp.greater(self.potential.value, self._tau).astype(self._dtype)
        # Reset neuron 
        self.potential.value = (1 - spikes) * self.potential.value
        # Set neuron refractory period.
        self._refractory.value = (1 - spikes) * (self._refractory.value + self._dt)
        return {
            'spikes': SpikeArray(spikes)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################