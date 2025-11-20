#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import jax
import jax.numpy as jnp
import dataclasses as dc
from spark.core.tracers import Tracer
from spark.core.payloads import SpikeArray, CurrentArray
from spark.core.variables import Variable, Constant
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.somas.base import Soma, SomaConfig
from spark.core.flax_imports import data
from spark.nn.initializers.base import Initializer

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

        Init:
            units: tuple[int]
            potential_rest: float | jax.Array
            potential_reset: float | jax.Array
            potential_tau: float | jax.Array
            resistance: float | jax.Array
            threshold: float | jax.Array

        Input:
            in_spikes: SpikeArray
            
        Output:
            out_spikes: SpikeArray

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
    def build(self, input_specs: dict[str, InputSpec]) -> None:
        super().build(input_specs)
        # Initialize variables.
        _potential_rest = self.config.potential_rest.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _potential_reset = self.config.potential_reset.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _potential_tau = self.config.potential_tau.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _resistance = self.config.resistance.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _threshold = self.config.threshold.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        # Membrane. Substract potential_rest to potential related terms to rebase potential at zero.
        self.potential_rest = Constant(_potential_rest, dtype=self._dtype)
        self.potential_reset = Constant(_potential_reset - _potential_rest, dtype=self._dtype)
        self.potential_scale = Constant(self._dt / _potential_tau, dtype=self._dtype)
        # Conductance.
        self.resistance = Constant(_resistance, dtype=self._dtype) # Current is in pA for stability
        # Threshold.
        self.threshold = Constant(_threshold - _potential_rest, dtype=self._dtype)

    def _update_states(self, current: CurrentArray) -> None:
        """
            Update neuron's soma states variables.
        """
        self.potential.value += self.potential_scale * (-self.potential.value + self.resistance * current.value)

    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        # Compute spikes.
        spikes = jnp.greater(self.potential.value, self.threshold.value).astype(self._dtype)
        # Reset neurons.
        self.potential.value = spikes * self.potential_reset + (1 - spikes) * self.potential.value
        return SpikeArray(spikes)
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class RefractoryLeakySomaConfig(LeakySomaConfig):
    """
        RefractoryLeakySoma model configuration class.
    """

    cooldown: float | jax.Array | Initializer = dc.field(
        default = 2.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Soma refractory period.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class RefractoryLeakySoma(LeakySoma):
    """
        Leaky soma with refractory time model.

        Init:
            units: tuple[int]
            potential_rest: float | jax.Array
            potential_reset: float | jax.Array
            potential_tau: float | jax.Array
            resistance: float | jax.Array
            threshold: float | jax.Array
            cooldown: float | jax.Array

        Input:
            in_spikes: SpikeArray
            
        Output:
            out_spikes: SpikeArray

        Reference: 
            Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. 
            Gerstner W, Kistler WM, Naud R, Paninski L. 
            Chapter 1.3 Integrate-And-Fire Models
            https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
    """
    config: RefractoryLeakySomaConfig

    def __init__(self, config: RefractoryLeakySomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    # NOTE: potential_rest is substracted to potential related terms to rebase potential at zero.
    def build(self, input_specs: dict[str, InputSpec]) -> None:
        super().build(input_specs)
        # Initialize variables.
        _cooldown = self.config.cooldown.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        # Refractory period.
        self.cooldown = Constant(jnp.round(_cooldown / self._dt).astype(jnp.uint16), dtype=jnp.uint16)
        self.refractory = Variable(self.cooldown * jnp.ones(self.units), dtype=jnp.uint16)
        self.is_ready = Variable(jnp.ones(self.units), dtype=jnp.bool)

    def reset(self) -> None:
        """
            Resets component state.
        """
        super().reset()
        self.refractory.value = jnp.array(self.cooldown, dtype=jnp.uint16)

    def _update_states(self, current: CurrentArray) -> None:
        """
            Update neuron's soma states variables.
        """
        self.is_ready.value = jnp.greater(self.refractory.value, self.cooldown)
        is_ready = self.is_ready.value.astype(self._dtype)
        self.potential.value += self.potential_scale * (-self.potential.value + is_ready * self.resistance * current.value)

    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        # Compute spikes.
        spikes = jnp.logical_and(
            jnp.greater(self.potential.value, self.threshold.value), 
            self.is_ready.value
        ).astype(self._dtype)
        # Reset neurons.
        self.potential.value = spikes * self.potential_reset + (1 - spikes) * self.potential.value
        # Set neuron refractory period.
        self.refractory.value = (1 - spikes).astype(jnp.uint16) * (self.refractory.value + 1)
        return SpikeArray(spikes)
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class StrictRefractoryLeakySomaConfig(RefractoryLeakySomaConfig):
    """
        StrictRefractoryLeakySoma model configuration class.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class StrictRefractoryLeakySoma(RefractoryLeakySoma):
    """
        Leaky soma with strict refractory time model. 
        Note: This model is here mostly for didactic/historical reasons.

        Init:
            units: tuple[int]
            potential_rest: float | jax.Array
            potential_reset: float | jax.Array
            potential_tau: float | jax.Array
            resistance: float | jax.Array
            threshold: float | jax.Array
            cooldown: float | jax.Array

        Input:
            in_spikes: SpikeArray
            
        Output:
            out_spikes: SpikeArray

        Reference: 
            Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. 
            Gerstner W, Kistler WM, Naud R, Paninski L. 
            Chapter 1.3 Integrate-And-Fire Models
            https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
    """
    config: StrictRefractoryLeakySomaConfig

    def _update_states(self, current: CurrentArray) -> None:
        """
            Update neuron's soma states variables.
        """
        self.is_ready.value = jnp.greater_equal(self.refractory.value, self.cooldown)
        is_ready = self.is_ready.value.astype(self._dtype)
        self.potential.value += is_ready * self.potential_scale * (-self.potential.value + self.resistance * current.value)
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class AdaptiveLeakySomaConfig(RefractoryLeakySomaConfig):
    """
        AdaptiveLeakySoma model configuration class.
    """

    threshold_tau: float | jax.Array | Initializer = dc.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Adaptive action potential threshold decay constant.',
        })
    threshold_delta: float | jax.Array | Initializer = dc.field(
        default = 100.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Adaptive action potential threshold after spike increment.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class AdaptiveLeakySoma(RefractoryLeakySoma):
    """
        Adaptive leaky soma model.

        Init:
            units: tuple[int]
            potential_rest: float | jax.Array
            potential_reset: float | jax.Array
            potential_tau: float | jax.Array
            resistance: float | jax.Array
            threshold: float | jax.Array
            cooldown: float | jax.Array
            threshold_tau: float | jax.Array
            threshold_delta: float | jax.Array

        Input:
            in_spikes: SpikeArray
            
        Output:
            out_spikes: SpikeArray

        Reference: 
            Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. 
            Gerstner W, Kistler WM, Naud R, Paninski L. 
            Chapter 5.1 Thresholds in a nonlinear integrate-and-fire model
            https://neuronaldynamics.epfl.ch/online/Ch5.S1.html
    """
    config: AdaptiveLeakySomaConfig

    def __init__(self, config: AdaptiveLeakySomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    # NOTE: potential_rest is substracted to potential related terms to rebase potential at zero.
    def build(self, input_specs: dict[str, InputSpec]) -> None:
        super().build(input_specs)
        # Initialize variables.
        _threshold_tau = self.config.threshold_tau.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _threshold_delta = self.config.threshold_delta.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        # Replace constant threshold with a tracer.
        self.threshold = data(Tracer(
            self.units,
            tau=_threshold_tau, 
            base=(self.threshold), 
            scale=_threshold_delta, 
            dt=self._dt, dtype=self._dtype
        ))

    def reset(self) -> None:
        """
            Resets component state.
        """
        super().reset()
        self.threshold.reset()

    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        # Compute spikes.
        spikes = super()._compute_spikes()
        # Update thresholds
        self.threshold(spikes.value)
        return spikes
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################