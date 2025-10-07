#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import jax.numpy as jnp
import dataclasses as dc
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.variables import Variable, Constant
from spark.core.shape import Shape
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.input.base import InputInterface, InputInterfaceConfig, InputInterfaceOutput

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class LinearSpikerConfig(InputInterfaceConfig):
    """
        LinearSpiker model configuration class.
    """
    
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

    def __init__(self, config: LinearSpikerConfig | None = None, **kwargs):
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

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################