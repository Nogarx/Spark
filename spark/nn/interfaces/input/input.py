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
import dataclasses as dc
import typing as tp
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.variables import Variable, Constant
from spark.core.shape import Shape
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator, BinaryValidator
from spark.nn.interfaces.base import Interface, InterfaceConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic InputInterface output contract.
class InputInterfaceOutput(tp.TypedDict):
    spikes: SpikeArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InputInterface(Interface, abc.ABC):
    """
        Abstract input interface model.
    """

    def __init__(self, config: InterfaceConfig = None, **kwargs):
        # Main attributes
        super().__init__(config = config, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args: SparkPayload, **kwargs) -> dict[str, SparkPayload]:
        """
            Transform the input signal into an Spike signal.
        """
        pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class PoissonSpikerConfig(InterfaceConfig):
    max_freq: float = dc.field(
        default = 50.0, 
        metadata = {
            'units': 'Hz',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Maximum firing frequency of the spiker.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class PoissonSpiker(InputInterface):
    """
        Transforms a continuous signal to a spiking signal.
        This transformation assumes a very simple poisson neuron model without any type of adaptation or plasticity.

        Init:
            max_freq: float [Hz]

        Input:
            signal: FloatArray

        Output:
            spikes: SpikeArray
    """
    config: PoissonSpikerConfig

    def __init__(self, config: PoissonSpikerConfig = None, **kwargs):
        # Initialize super
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.max_freq = self.config.max_freq
        self._scale = self._dt * (self.max_freq / 1000)

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Initialize shapes
        self._shape = Shape(input_specs['signal'].shape)

    def __call__(self, signal: FloatArray) -> InputInterfaceOutput:
        """
            Input interface operation.

            Input: 
                A FloatArray of values in the range [0,1].
            Output: 
                A SpikeArray of the same shape as the input.
        """
        spikes = (jax.random.uniform(self.get_rng_keys(1), shape=self._shape) < self._scale * signal.value).astype(self._dtype)
        return {
            'spikes': SpikeArray(spikes)
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class LinearSpikerConfig(InterfaceConfig):
    tau: float = dc.field(
        default = 100.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Decay time constant of the membrane potential of the units of the spiker.',
        })
    cd: float = dc.field(
        default = 2.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': '(Cooldown) Refractory period of the units of the spiker.',
        })
    max_freq: float = dc.field(
        default = 50.0, 
        metadata = {
            'units': 'Hz',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Maximum firing frequency of the spiker.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class LinearSpiker(InputInterface):
    """
        Transforms a continuous signal to a spiking signal.
        This transformation assumes a very simple linear neuron model without any type of adaptation or plasticity.
        Units have a fixed refractory period and at maximum input signal will fire up to some fixed frequency.

        Init:
            tau: float [ms]
            cd: float [ms]
            max_freq: float [Hz]

        Input:
            signal: FloatArray
            
        Output:
            spikes: SpikeArray
    """
    config: LinearSpikerConfig

    def __init__(self, config: LinearSpikerConfig = None, **kwargs):
        # Initialize super
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.tau = self.config.tau
        self.cd = self.config.cd
        self.max_freq = self.config.max_freq
        exp_term = jnp.exp((1/self.tau) * ((1000-self.cd*self.max_freq) / self.max_freq)) # dt cancels out
        scale = ((1 / (exp_term - 1)) + 1)
        self._tau = Constant(self.tau, dtype=self._dtype)
        self._scale = Constant(scale, dtype=self._dtype)
        self._decay = Constant(jnp.exp(-self._dt / self.tau), dtype=self._dtype)
        self._gain = Constant(1 - self._decay, dtype=self._dtype)

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Initialize shapes
        self._shape = Shape(input_specs['signal'].shape)
        # Initialize variables
        self._cooldown = Constant(self.cd * jnp.ones(shape=self._shape), dtype=self._dtype)
        self._refractory = Variable(self._cooldown, dtype=self._dtype)
        self.potential = Variable(jnp.zeros(shape=self._shape), dtype=self._dtype)

    def reset(self,):
        """
            Reset module to its default state.
        """
        self.potential.value = jnp.zeros(shape=self._shape)
        self._refractory.value = self._cooldown.value

    def __call__(self, signal: FloatArray) -> InputInterfaceOutput:
        """
            Input interface operation.

            Input: 
                A FloatArray of values in the range [0,1].
            Output: 
                A SpikeArray of the same shape as the input.
        """
        # Update potential
        is_ready = jnp.greater_equal(self._refractory.value, self._cooldown).astype(self._dtype)
        dV = is_ready * self._tau * self._gain * self._scale * signal.value
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

@register_config
class TopologicalSpikerConfig(InterfaceConfig):
    glue: jax.Array = dc.field(
        default_factory = lambda: jnp.array(0), 
        metadata = {
            'validators': [
                TypeValidator,
                BinaryValidator,
            ],
            'description': 'Jax array indicating if the borders of the cube are glued together. \
                            Entries must be either one or zero, indicating gluing and not gluing, respectively. \
                            It may be either an array with a single element or \
                            an array with the same dimensionality as the input vector.',
        })
    mins: jax.Array = dc.field(
        default_factory = lambda: jnp.array(0), 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Minimum value for the rescaling factor. It may be either an array with a single element or \
                            an array with the same dimensionality as the input vector.',
        })
    maxs: jax.Array = dc.field(
        default_factory = lambda: jnp.array(1), 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Maximum value for the rescaling factor. It may be either an array with a single element or \
                            an array with the same dimensionality as the input vector.',
        })
    resolution: int = dc.field(
        default = 64, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Virtual units subdivision of the space per dimension.',
        })
    sigma: float = dc.field(
        default = 1/32, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Spreed of the signal (standard deviation of a gaussian) in target manifold.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class TopologicalPoissonSpikerConfig(TopologicalSpikerConfig, PoissonSpikerConfig):
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class TopologicalPoissonSpiker(InputInterface):
    """
        Transforms a continuous signal to a spiking signal.
        This transformation maps a vector (a point in a hypercube) into a simple manifold with/without its borders glued.
        This transformation assumes a very simple poisson neuron model without any type of adaptation or plasticity.

        Init:
            glue: jax.Array
            mins: jax.Array
            maxs: jax.Array
            resolution: int 
            max_freq: float [Hz]
            sigma: float

        Input:
            signal: FloatArray
            
        Output:
            spikes: SpikeArray
    """
    config: TopologicalPoissonSpikerConfig

    def __init__(self, config: TopologicalPoissonSpikerConfig = None, **kwargs):
        # Initialize super
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.resolution = self.config.resolution
        self.max_freq = self.config.max_freq
        self.sigma = self.config.sigma
        self._scale = self._dt * (self.max_freq / 1000)
        self._glue = Constant(self.config.glue, dtype=jnp.bool_)
        self._mins = Constant(self.config.mins, dtype=self._dtype)
        self._maxs = Constant(self.config.maxs, dtype=self._dtype)
        self._sigma = Constant(self.sigma, dtype=self._dtype)


    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Initialize shapes
        input_shape = Shape(input_specs['signal'].shape)
        self._output_shape = Shape(input_specs['signal'].shape + (self.resolution,))
        # Initialize variables
        self._space = Constant(jnp.linspace(jnp.zeros(input_shape), 
                                                 jnp.pi*jnp.ones(input_shape), 
                                                 self.resolution), 
                                    dtype=self._dtype)
        
    def __call__(self, signal: FloatArray) -> InputInterfaceOutput:
        """
            Input interface operation.

            Input: A FloatArray of values in the range [mins, maxs].
            Output: A SpikeArray of the same shape as the input.
        """
        # Transform input to [0, 1]
        x = (signal.value - self._mins) / (self._maxs - self._mins)
        x = jnp.where(self._glue.value, jnp.sin(self._space + x*jnp.pi), jnp.tanh(self._space - x*jnp.pi))
        x = jnp.exp( -(0.5 / self._sigma) * (x**2) ).T
        # Poisson process
        spikes = (jax.random.uniform(self.get_rng_keys(1), shape=self._output_shape) < self._scale * x).astype(self._dtype)
        return {
            'spikes': SpikeArray(spikes)
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class TopologicalLinearSpikerConfig(TopologicalSpikerConfig, LinearSpikerConfig):
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class TopologicalLinearSpiker(InputInterface):
    """
        Transforms a continuous signal to a spiking signal.
        This transformation maps a vector (a point in a hypercube) into a simple manifold with/without its borders glued.
        This transformation assumes a very simple linear neuron model without any type of adaptation or plasticity.

        Init:
            glue: jax.Array
            mins: jax.Array
            maxs: jax.Array
            resolution: int 
            tau: float [ms]
            cd: float [ms]
            max_freq: float [Hz]
            sigma: float

        Input:
            signal: FloatArray
            
        Output:
            spikes: SpikeArray
    """
    config: TopologicalLinearSpikerConfig

    def __init__(self, config: TopologicalLinearSpikerConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.resolution = self.config.resolution
        self.tau = self.config.tau
        self.cd = self.config.cd
        self.max_freq = self.config.max_freq
        self.sigma = self.config.sigma
        self._glue = Constant(self.config.glue, dtype=jnp.bool_)
        self._mins = Constant(self.config.mins, dtype=self._dtype)
        self._maxs = Constant(self.config.maxs, dtype=self._dtype)
        self._sigma = Constant(self.sigma, dtype=self._dtype)
        exp_term = jnp.exp((1/self.tau) * ((1000-self.cd*self.max_freq) / self.max_freq)) # dt cancels out
        scale = ((1 / (exp_term - 1)) + 1)
        self._scale = Constant(scale, dtype=self._dtype)
        self._tau = Constant(self.tau, dtype=self._dtype)
        self._decay = Constant(jnp.exp(-self._dt / self._tau), dtype=self._dtype)
        self._gain = Constant(1 - self._decay, dtype=self._dtype)

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Initialize shapes
        input_shape = Shape(input_specs['signal'].shape)
        self._output_shape = Shape(input_specs['signal'].shape + (self.resolution,))
        # Initialize variables
        self._space = Constant(jnp.linspace(jnp.zeros(input_shape), 
                                                 jnp.pi*jnp.ones(input_shape), 
                                                 self.resolution), 
                                    dtype=self._dtype)
        self._cooldown = Constant(self.cd * jnp.ones(shape=self._output_shape), dtype=self._dtype)
        self._refractory = Variable(self._cooldown, dtype=self._dtype)
        self.potential = Variable(jnp.zeros(shape=self._output_shape), dtype=self._dtype)

    def reset(self,):
        """
            Reset module to its default state.
        """
        self.potential.value = jnp.zeros(shape=self._output_shape)
        self._refractory.value = self._cooldown.value

    def __call__(self, signal: FloatArray) -> InputInterfaceOutput:
        """
            Input interface operation.

            Input: A FloatArray of values in the range [mins, maxs].
            Output: A SpikeArray of the same shape as the input.
        """
        # Transform input to [0, 1]
        x = (signal.value - self._mins) / (self._maxs - self._mins)
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