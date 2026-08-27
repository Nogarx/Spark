#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs

import jax.numpy as jnp
import dataclasses as dc
import spark.core.utils as utils
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.backend import Variable, Constant
from spark.core.registry import register_interface, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.input.base import InputInterface, InputInterfaceConfig, InputInterfaceOutput

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class LinearSpikerConfig(InputInterfaceConfig):
    """
        Configuration for `LinearSpiker`.

        Parameters
        ----------
        tau : float, default 100.0
            Decay constant of the internal potential, in ms.
        cd : float, default 2.0
            Refractory period, in ms. Bounds the firing rate at ``1000 / cd`` Hz.
        max_freq : float, default 100.0
            Firing rate reached by an input of 1.0, in Hz.
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
        default = 100.0, 
        metadata = {
            'units': 'Hz',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Maximum firing frequency of the spiker.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class LinearSpiker(InputInterface):
    """
        Deterministic rate encoding of a continuous signal.

        Each unit integrates its input into a potential and fires when the potential reaches a
        threshold, after which it resets and waits out a refractory period. The same input always
        produces the same spike train, which makes a run repeatable without fixing a seed.

        Parameters
        ----------
        config : LinearSpikerConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        signal : FloatArray
            Value to encode, expected in ``[0, 1]``.

        Output Ports
        ------------
        spikes : SpikeArray
            One spike train per input, of the same shape as ``signal``.

        Notes
        -----
        The input is gated off during the refractory period, so ``cd`` bounds the firing rate at
        ``1000 / cd`` Hz regardless of ``max_freq``.

        See Also
        --------
        PoissonSpiker : Stochastic encoding of the same signal.
    """
    config: LinearSpikerConfig

    def __init__(self, config: LinearSpikerConfig | None = None, **kwargs):
        # Initialize super
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.tau = self.config.tau
        self.cd = self.config.cd
        self.max_freq = self.config.max_freq
        exp_term = jnp.exp((self._dt/self.tau) * ((1000-self.cd*self.max_freq) / self.max_freq)) # dt cancels out
        scale = ((1 / (exp_term - 1)) + 1)
        self._tau = Constant(self.tau, dtype=self._dtype)
        self._scale = Constant(scale, dtype=self._dtype)
        self._decay = Constant(jnp.exp(-self._dt / self.tau), dtype=self._dtype)
        self._gain = Constant(1 - self._decay.value, dtype=self._dtype)

    def build(self, signal: FloatArray) -> None:
        # Initialize shapes
        self._shape = utils.validate_shape(signal.shape)
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
            Encodes the signal as spikes.

            Parameters
            ----------
            signal : FloatArray
                Value to encode, expected in ``[0, 1]``.

            Returns
            -------
            InputInterfaceOutput
                Dictionary with one entry, ``spikes``, of the same shape as ``signal``.
        """
        # Update potential
        is_ready = jnp.greater_equal(self._refractory.value, self._cooldown).astype(self._dtype)
        dV = is_ready * self._tau.value * self._gain.value * self._scale.value * signal.value
        self.potential.value = self._decay.value * self.potential.value + self._dt * dV
        # Spike
        spikes = (self.potential.value > self._tau.value).astype(self._dtype)
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