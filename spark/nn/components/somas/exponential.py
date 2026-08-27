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
from spark.core.payloads import SparkPayload
from spark.core.backend import Constant
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.somas.base import Soma, SomaConfig
from spark.nn.initializers.base import Initializer
from spark.nn.components.somas.adaptive import AdaptiveSoma, AdaptiveSomaConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ExponentialSomaConfig(SomaConfig):
    """
        ExponentialSoma model configuration class.
    """
    potential_rest: float | jax.Array | Initializer = dc.field(
        default = -70.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane rest potential.',
        })
    potential_reset: float | jax.Array | Initializer = dc.field(
        default = -51.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane after spike reset potential.',
        })
    potential_tau: float | jax.Array | Initializer = dc.field(
        default = 5.0,
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Membrane potential decay constant.',
        })
    resistance: float | jax.Array | Initializer = dc.field(
        default = 0.5,
        metadata = {
            'units': 'GΩ', # [1/nS]
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane resistance.',
        })
    threshold: float | jax.Array | Initializer = dc.field(
        default = -30.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Action potential threshold base value.',
        })
    rheobase_threshold: float | jax.Array | Initializer = dc.field(
        default = -50.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Rheobase threshold (exponential term threshold).',
        })
    spike_slope: float | jax.Array | Initializer = dc.field(
        default = 2.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Sharpness of action potential initiation.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ExponentialSoma(Soma):
    """
        Exponential soma model.

        Refractoriness and adaptation are not part of this model. The adaptive exponential
        (AdEx) model is AdaptiveExponentialSoma, this model composed with the adaptation
        extension and given an adaptation current.

        Init:
            units: tuple[int, ...]
            potential_rest: float | jax.Array
            potential_reset: float | jax.Array
            potential_tau: float | jax.Array
            resistance: float | jax.Array
            threshold: float | jax.Array
            rheobase_threshold: float | jax.Array
            spike_slope: float | jax.Array

        Input:
            current: CurrentArray

        Output:
            spikes: SpikeArray

        Reference:
            How Spike Generation Mechanisms Determine the Neuronal Response to Fluctuating Inputs
            Nicolas Fourcaud-Trocmé, David Hansel, Carl van Vreeswijk, and Nicolas Brunel
            The Journal of Neuroscience, December 17, 2003
            https://www.jneurosci.org/content/23/37/11628
            Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
            Gerstner W, Kistler WM, Naud R, Paninski L.
            Chapter 5.2 Exponential Integrate-and-Fire Model
            https://neuronaldynamics.epfl.ch/online/Ch5.S2.html
    """
    config: ExponentialSomaConfig

    def __init__(self, config: ExponentialSomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    # NOTE: potential_rest is substracted to potential related terms to rebase potential at zero.
    def build(self, **abc_args: SparkPayload) -> None:
        super().build(**abc_args)
        # Initialize variables.
        _potential_rest = self.config.init.potential_rest(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _potential_reset = self.config.init.potential_reset(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _potential_tau = self.config.init.potential_tau(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _resistance = self.config.init.resistance(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _threshold = self.config.init.threshold(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _rheobase_threshold = self.config.init.rheobase_threshold(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _spike_slope = self.config.init.spike_slope(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        # Membrane. Substract potential_rest to potential related terms to rebase potential at zero.
        self.potential_rest = Constant(_potential_rest, dtype=self._dtype)
        self.potential_reset = Constant(_potential_reset - _potential_rest, dtype=self._dtype)
        self.potential_scale = Constant(self._dt / _potential_tau, dtype=self._dtype)
        # Conductance.
        self.resistance = Constant(_resistance, dtype=self._dtype)
        # Threshold.
        self.threshold = Constant(_threshold - _potential_rest, dtype=self._dtype)
        self.rheobase_threshold = Constant(_rheobase_threshold - _potential_rest, dtype=self._dtype)
        # Spike slope.
        self.spike_slope = Constant(_spike_slope, dtype=self._dtype)

    def _integrate(self, potential: jax.Array, current: jax.Array) -> jax.Array:
        """
            Membrane integration.
        """
        return potential + self.potential_scale.value * (
            - potential
            + self.spike_slope.value * jnp.exp((potential - self.rheobase_threshold.value) / self.spike_slope.value)
            + self.resistance.value * current
        )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class AdaptiveExponentialSomaConfig(AdaptiveSomaConfig, ExponentialSomaConfig):
    """
        AdaptiveExponentialSoma model configuration class.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class AdaptiveExponentialSoma(AdaptiveSoma, ExponentialSoma):
    """
        Exponential soma model with the adaptation extension.

        Input:
            current: CurrentArray

        Output:
            spikes: SpikeArray
    """
    config: AdaptiveExponentialSomaConfig

    def __init__(self, config: AdaptiveExponentialSomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################