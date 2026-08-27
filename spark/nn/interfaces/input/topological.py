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
import spark.core.utils as utils
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.backend import Variable, Constant
from spark.core.registry import register_interface, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator, BinaryValidator
from spark.nn.interfaces.input.base import InputInterface, InputInterfaceConfig, InputInterfaceOutput
from spark.nn.interfaces.input.poisson import PoissonSpikerConfig
from spark.nn.interfaces.input.linear import LinearSpikerConfig
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class TopologicalSpikerConfig(InputInterfaceConfig):
    """
        Base configuration for topological spikers.

        Parameters
        ----------
        glue : jax.Array, default 0
            Per-dimension flag marking whether the two ends of that dimension are identified. A
            glued dimension is encoded on a circle, so its two extremes excite the same units.
        mins : jax.Array, default 0
            Lower end of the input range, per dimension or as a single value.
        maxs : jax.Array, default 1
            Upper end of the input range, per dimension or as a single value.
        resolution : int, default 64
            Number of units each input dimension is spread over.
        sigma : float, default 1/32
            Width of the activity bump, as the standard deviation of a Gaussian in the target
            space.
    """
    
    glue: int | jax.Array | Initializer = dc.field(
        default = 0, 
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
    mins: int | jax.Array | Initializer = dc.field(
        default = 0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Minimum value for the rescaling factor. It may be either an array with a single element or \
                            an array with the same dimensionality as the input vector.',
        })
    maxs: int | jax.Array | Initializer = dc.field(
        default = 1, 
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
    """
        Configuration for `TopologicalPoissonSpiker`.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class TopologicalPoissonSpiker(InputInterface):
    """
        Place-coded stochastic encoding of a continuous signal.

        Each input dimension is spread over ``resolution`` units laid out along that dimension.
        A value excites the units near its position under a Gaussian bump of width ``sigma``, and
        those units then fire as independent Poisson processes. Nearby values excite overlapping
        populations, which a per-unit rate code does not give.

        Setting ``glue`` for a dimension identifies its two ends, so that dimension is encoded on
        a circle. (e.g. an angular signal should reconciliate units placed at 0 and 2π positions)

        Parameters
        ----------
        config : TopologicalPoissonSpikerConfig
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        signal : FloatArray
            Value to encode, expected in ``[mins, maxs]``.

        Output Ports
        ------------
        spikes : SpikeArray
            Place-coded spike trains, of shape ``signal.shape + (resolution,)``.

        See Also
        --------
        PoissonSpiker : One unit per input, without the place code.
        TopologicalLinearSpiker : The same place code, deterministically encoded.
    """
    config: TopologicalPoissonSpikerConfig

    def __init__(self, config: TopologicalPoissonSpikerConfig | None = None, **kwargs):
        # Initialize super
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.resolution = self.config.resolution
        self.max_freq = self.config.max_freq
        self.sigma = self.config.sigma
        self._scale = self._dt * (self.max_freq / 1000)
        self._glue = Constant(self.config.init.glue(), dtype=jnp.bool_)
        self._mins = Constant(self.config.init.mins(), dtype=self._dtype)
        self._maxs = Constant(self.config.init.maxs(), dtype=self._dtype)
        self._sigma = Constant(self.sigma, dtype=self._dtype)


    def build(self, signal: FloatArray) -> None:
        # Initialize shapes
        input_shape = utils.validate_shape(signal.shape)
        self._output_shape = utils.validate_shape(signal.shape + (self.resolution,))
        # Initialize variables
        self._space = Constant(jnp.linspace(jnp.zeros(input_shape), 
                                                 jnp.pi*jnp.ones(input_shape), 
                                                 self.resolution), 
                                    dtype=self._dtype)
        
    def __call__(self, signal: FloatArray) -> InputInterfaceOutput:
        """
            Encodes the signal as spikes.

            Parameters
            ----------
            signal : FloatArray
                Value to encode, expected in ``[mins, maxs]``.

            Returns
            -------
            InputInterfaceOutput
                Dictionary with one entry, ``spikes``, of shape ``signal.shape + (resolution,)``.
        """
        # Transform input to [0, 1]
        x = (signal.value - self._mins.value) / (self._maxs.value - self._mins.value)
        x = jnp.where(self._glue.value, jnp.sin(self._space.value + x*jnp.pi), jnp.tanh(self._space.value - x*jnp.pi))
        x = jnp.exp( -(0.5 / self._sigma.value) * (x**2) ).T
        # Poisson process
        spikes = (jax.random.uniform(self.get_rng_keys(1), shape=self._output_shape) < self._scale * x).astype(self._dtype)
        return {
            'spikes': SpikeArray(spikes)
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class TopologicalLinearSpikerConfig(TopologicalSpikerConfig, LinearSpikerConfig):
    """
        Configuration for `TopologicalLinearSpiker`.

        Union of `TopologicalSpikerConfig` and `LinearSpikerConfig`. It declares no field of its
        own.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class TopologicalLinearSpiker(InputInterface):
    """
        Place-coded deterministic encoding of a continuous signal.

        Each input dimension is spread over ``resolution`` units laid out along that dimension.
        A value excites the units near its position under a Gaussian bump of width ``sigma``, and
        those units then fire as independent Poisson processes. Nearby values excite overlapping
        populations, which a per-unit rate code does not give.

        Setting ``glue`` for a dimension identifies its two ends, so that dimension is encoded on
        a circle. (e.g. an angular signal should reconciliate units placed at 0 and 2π positions)

        Parameters
        ----------
        config : TopologicalLinearSpikerConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        signal : FloatArray
            Value to encode, expected in ``[mins, maxs]``.

        Output Ports
        ------------
        spikes : SpikeArray
            Place-coded spike trains, of shape ``signal.shape + (resolution,)``.

        See Also
        --------
        TopologicalPoissonSpiker : The same place code, stochastically encoded.
        LinearSpiker : One unit per input, without the place code.
    """
    config: TopologicalLinearSpikerConfig

    def __init__(self, config: TopologicalLinearSpikerConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.resolution = self.config.resolution
        self.tau = self.config.tau
        self.cd = self.config.cd
        self.max_freq = self.config.max_freq
        self.sigma = self.config.sigma
        self._glue = Constant(self.config.init.glue(), dtype=jnp.bool_)
        self._mins = Constant(self.config.init.mins(), dtype=self._dtype)
        self._maxs = Constant(self.config.init.maxs(), dtype=self._dtype)
        self._sigma = Constant(self.sigma, dtype=self._dtype)
        exp_term = jnp.exp((1/self.tau) * ((1000-self.cd*self.max_freq) / self.max_freq)) # dt cancels out
        scale = ((1 / (exp_term - 1)) + 1)
        self._scale = Constant(scale, dtype=self._dtype)
        self._tau = Constant(self.tau, dtype=self._dtype)
        self._decay = Constant(jnp.exp(-self._dt / self._tau.value), dtype=self._dtype)
        self._gain = Constant(1 - self._decay.value, dtype=self._dtype)

    def build(self, signal: FloatArray) -> None:
        # Initialize shapes
        input_shape = utils.validate_shape(signal.shape)
        self._output_shape = utils.validate_shape(signal.shape + (self.resolution,))
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
            Encodes the signal as spikes.

            Parameters
            ----------
            signal : FloatArray
                Value to encode, expected in ``[mins, maxs]``.

            Returns
            -------
            InputInterfaceOutput
                Dictionary with one entry, ``spikes``, of shape ``signal.shape + (resolution,)``.
        """
        # Transform input to [0, 1]
        x = (signal.value - self._mins.value) / (self._maxs.value - self._mins.value)
        x = jnp.where(self._glue.value, jnp.sin(self._space.value + x*jnp.pi), jnp.tanh(self._space.value - x*jnp.pi))
        x = jnp.exp( -(0.5 / self._sigma.value) * (x**2) ).T
        # Update potential. Note: dt cancels out.
        is_ready = jnp.greater_equal(self._refractory.value, self._cooldown).astype(self._dtype)
        dV = is_ready * self._tau.value * self._gain.value * self._scale.value * x
        self.potential.value = self._decay.value * self.potential.value + dV
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