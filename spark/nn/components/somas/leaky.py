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
class LeakySomaConfig(SomaConfig):
    """
        LeakySoma model configuration class.
    """

    potential_rest: float | jax.Array | Initializer = dc.field(
        default = -60.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane rest potential.',
        })
    potential_reset: float | jax.Array | Initializer = dc.field(
        default = -50.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane after spike reset potential.',
        })
    potential_tau: float | jax.Array | Initializer = dc.field(
        default = 20.0,
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Membrane potential decay constant.',
        })
    resistance: float | jax.Array | Initializer = dc.field(
        default = 0.1,
        metadata = {
            'units': 'GΩ', # [1/nS]
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane resistance.',
        })
    threshold: float | jax.Array | Initializer = dc.field(
        default = -40.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Action potential threshold base value.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class LeakySoma(Soma):
    """
        Leaky soma model.

        Refractoriness and threshold adaptation are not part of this model.
        AdaptiveLeakySoma is this model composed with the adaptation extension and is what
        provides them.

        Init:
            units: tuple[int, ...]
            potential_rest: float | jax.Array
            potential_reset: float | jax.Array
            potential_tau: float | jax.Array
            resistance: float | jax.Array
            threshold: float | jax.Array

        Input:
            current: CurrentArray

        Output:
            spikes: SpikeArray

        Reference:
            Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
            Gerstner W, Kistler WM, Naud R, Paninski L.
            Chapter 1.3 Integrate-And-Fire Models
            https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
    """
    config: LeakySomaConfig

    def __init__(self, config: LeakySomaConfig | None = None, **kwargs) -> None:
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
        # Membrane. Substract potential_rest to potential related terms to rebase potential at zero.
        self.potential_rest = Constant(_potential_rest, dtype=self._dtype)
        self.potential_reset = Constant(_potential_reset - _potential_rest, dtype=self._dtype)
        self.potential_decay = Constant(jnp.exp(-self._dt / _potential_tau), dtype=self._dtype)
        self.potential_gain = Constant((1 - self.potential_decay.value), dtype=self._dtype)
        # Conductance.
        self.resistance = Constant(_resistance, dtype=self._dtype) # Current is in pA for stability
        # Threshold.
        self.threshold = Constant(_threshold - _potential_rest, dtype=self._dtype)

    def _integrate(self, potential: jax.Array, current: jax.Array) -> jax.Array:
        """
            Membrane integration.
        """
        return (
            + self.potential_decay.value * potential
            + self.potential_gain.value * self.resistance.value * current
        )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class AdaptiveLeakySomaConfig(AdaptiveSomaConfig, LeakySomaConfig):
    """
        AdaptiveLeakySoma model configuration class.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class AdaptiveLeakySoma(AdaptiveSoma, LeakySoma):
    """
        Leaky soma model with the adaptation extension.

        Input:
            current: CurrentArray

        Output:
            spikes: SpikeArray
    """
    config: AdaptiveLeakySomaConfig

    def __init__(self, config: AdaptiveLeakySomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################