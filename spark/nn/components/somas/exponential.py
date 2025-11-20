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
from spark.core.flax_imports import data
from spark.nn.components.somas.base import Soma, SomaConfig
from spark.nn.initializers.base import Initializer

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
    def build(self, input_specs: dict[str, InputSpec]) -> None:
        super().build(input_specs)
        # Initialize variables.
        _potential_rest = self.config.potential_rest.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _potential_reset = self.config.potential_reset.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _potential_tau = self.config.potential_tau.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _resistance = self.config.resistance.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _threshold = self.config.threshold.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _rheobase_threshold = self.config.rheobase_threshold.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _spike_slope = self.config.spike_slope.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
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

    def _update_states(self, current: CurrentArray) -> None:
        """
            Update neuron's soma states variables.
        """
        self.potential.value += self.potential_scale * (
            - self.potential.value
            + self.spike_slope * jnp.exp((self.potential.value - self.rheobase_threshold.value)/self.spike_slope)
            + self.resistance * current.value
        )

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
class RefractoryExponentialSomaConfig(ExponentialSomaConfig):
    """
        RefractoryExponentialSoma model configuration class.
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
class RefractoryExponentialSoma(ExponentialSoma):
    """
        Exponential soma with refractory time model.

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
            How Spike Generation Mechanisms Determine the Neuronal Response to Fluctuating Inputs
            Nicolas Fourcaud-Trocmé, David Hansel, Carl van Vreeswijk, and Nicolas Brunel
            The Journal of Neuroscience, December 17, 2003
            https://www.jneurosci.org/content/23/37/11628
            Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. 
            Gerstner W, Kistler WM, Naud R, Paninski L. 
            Chapter 5.2 Exponential Integrate-and-Fire Model
            https://neuronaldynamics.epfl.ch/online/Ch5.S2.html
    """
    config: RefractoryExponentialSomaConfig

    def __init__(self, config: RefractoryExponentialSomaConfig | None = None, **kwargs) -> None:
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
        self.refractory.value = jnp.array(self.cooldown, dtype=self._dtype)

    def _update_states(self, current: CurrentArray) -> None:
        """
            Update neuron's soma states variables.
        """
        self.is_ready.value = jnp.greater(self.refractory.value, self.cooldown)
        is_ready = self.is_ready.value.astype(self._dtype)
        
        self.potential.value += self.potential_scale * (
            -self.potential.value + 
            self.spike_slope * jnp.exp((self.potential.value - self.rheobase_threshold.value)/self.spike_slope) +
            is_ready * self.resistance * current.value
        )

    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        # Compute spikes.
        spikes = jnp.logical_and(
            jnp.greater(self.potential.value, self.threshold.value), 
            jnp.greater(self.refractory.value, self.cooldown)
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
class AdaptiveExponentialSomaConfig(ExponentialSomaConfig):
    """
        AdaptiveExponentialSoma model configuration class.
    """

    adaptation_tau: float | jax.Array | Initializer = dc.field(
        default = 100.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Adaptation current decay constant.',
        })
    adaptation_delta: float | jax.Array | Initializer = dc.field(
        default = 7.0, 
        metadata = {
            'units': 'pA',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Adaptation current after spike increment.',
        })
    adaptation_subthreshold: float | jax.Array | Initializer = dc.field(
        default = 0.5, 
        metadata = {
            'units': 'nS', # 1/GΩ
            'validators': [
                TypeValidator,
            ], 
            'description': 'Scale factor of the subthreshold adaptation (potential-based adaptation).',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class AdaptiveExponentialSoma(ExponentialSoma):
    """
        Adaptive Exponential soma model.

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

            Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity.
            Romain Brette and Gerstner Wulfram
            Gerstner W, Kistler WM, Naud R, Paninski L. 
            Journal of Neurophysiology vol. 94, no. 5, pp. 3637-3642, 2005
            https://doi.org/10.1152/jn.00686.2005
            Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. 
            Gerstner W, Kistler WM, Naud R, Paninski L. 
            Chapter 5.2 Exponential Integrate-and-Fire Model
            https://neuronaldynamics.epfl.ch/online/Ch5.S2.html
    """
    config: AdaptiveExponentialSomaConfig

    def __init__(self, config: AdaptiveExponentialSomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    # NOTE: potential_rest is substracted to potential related terms to rebase potential at zero.
    def build(self, input_specs: dict[str, InputSpec]) -> None:
        super().build(input_specs)
        # Initialize variables.
        _adaptation_delta = self.config.adaptation_delta.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _adaptation_subthreshold = self.config.adaptation_subthreshold.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _adaptation_tau = self.config.adaptation_tau.init(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        # Overwrite constant threshold with a tracer.
        self.adaptation = Variable(jnp.zeros(self.units, dtype=self._dtype), dtype=self._dtype)
        self.adaptation_delta = Constant(_adaptation_delta, dtype=self._dtype)
        self.adaptation_subthreshold = Constant(_adaptation_subthreshold, dtype=self._dtype)
        self.adaptation_scale = Constant(self._dt / _adaptation_tau, dtype=self._dtype)

    def reset(self, ):
        super().reset()
        self.adaptation.value = jnp.zeros(self.units, dtype=self._dtype)

    def _update_states(self, current: CurrentArray) -> None:
        """
            Update neuron's soma states variables.
        """
        self.potential.value += self.potential_scale * (
            - self.potential.value
            + self.spike_slope * jnp.exp((self.potential.value - self.rheobase_threshold.value)/self.spike_slope)
            - self.resistance * self.adaptation.value
            + self.resistance * current.value
        )


    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        # Compute spikes.
        spikes = super()._compute_spikes()
        # Update adaptation
        self.adaptation.value += self.adaptation_scale * (
            - self.adaptation.value 
            + self.adaptation_subthreshold * self.potential.value
        ) + self.adaptation_delta * spikes.value.astype(self._dtype)
        return spikes
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class SimplifiedAdaptiveExponentialSomaConfig(RefractoryExponentialSomaConfig):
    """
        SimplifiedAdaptiveExponentialSoma model configuration class.
    """

    threshold_tau: float | jax.Array  | Initializer = dc.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Adaptive action potential threshold decay constant.',
        })
    threshold_delta: float | jax.Array  | Initializer = dc.field(
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
class SimplifiedAdaptiveExponentialSoma(RefractoryExponentialSoma):
    """
        Simplified Adaptive Exponential soma model. This model drops the subthreshold adaptation.

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

            Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity.
            Romain Brette and Gerstner Wulfram
            Gerstner W, Kistler WM, Naud R, Paninski L. 
            Journal of Neurophysiology vol. 94, no. 5, pp. 3637-3642, 2005
            https://doi.org/10.1152/jn.00686.2005
            Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. 
            Gerstner W, Kistler WM, Naud R, Paninski L. 
            Chapter 5.2 Exponential Integrate-and-Fire Model
            https://neuronaldynamics.epfl.ch/online/Ch5.S2.html
    """
    config: SimplifiedAdaptiveExponentialSomaConfig

    def __init__(self, config: SimplifiedAdaptiveExponentialSomaConfig | None = None, **kwargs) -> None:
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