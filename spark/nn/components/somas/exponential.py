#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import jax.numpy as jnp
import dataclasses as dc
from spark.core.tracers import Tracer
from spark.core.payloads import SpikeArray, CurrentArray
from spark.core.variables import Variable, Constant
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.somas.base import Soma, SomaConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ExponentialSomaConfig(SomaConfig):
    potential_rest: float = dc.field(
        default = -60.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Membrane rest potential.',
        })
    potential_reset: float = dc.field(
        default = -50.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Membrane after spike reset potential.',
        })
    potential_tau: float = dc.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Membrane potential decay constant.',
        })
    resistance: float = dc.field(
        default = 100.0,
        metadata = {
            'units': 'MΩ', # [1/µS]
            'validators': [
                TypeValidator,
            ], 
            'description': 'Membrane resistance.',
        })
    threshold: float = dc.field(
        default = -40.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Action potential threshold base value.',
        })
    spike_slope: float = dc.field(
        default = -40.0, 
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
            units: Shape
            potential_rest: float
            potential_reset: float
            potential_tau: float
            resistance: float
            threshold: float

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
            Chapter 1.3 Integrate-And-Fire Models
            https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
    """
    config: ExponentialSomaConfig

    # Type hints
    potential_reset: Constant
    potential_scale: Constant
    resistance: Constant
    threshold: Constant
    spike_slope: Constant

    def __init__(self, config: ExponentialSomaConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    # NOTE: potential_rest is substracted to potential related terms to rebase potential at zero.
    def build(self, input_specs: dict[str, InputSpec]):
        super().build(input_specs)
        # Initialize variables.
        # Membrane. Substract potential_rest to potential related terms to rebase potential at zero.
        self.potential_reset = Constant(self.config.potential_reset - self.config.potential_rest, dtype=self._dtype)
        self.potential_scale = Constant(self._dt / self.config.potential_tau, dtype=self._dtype)
        # Conductance.
        self.resistance = Constant(self.config.resistance / 1000.0, dtype=self._dtype) # Current is in pA for stability
        # Threshold.
        self.threshold = Constant(self.config.threshold - self.config.potential_rest, dtype=self._dtype)
        # Spike slope.
        self.spike_slope = Constant(self.config.spike_slope, dtype=self._dtype)

    def reset(self) -> None:
        """
            Resets component state.
        """
        self._potential.value = jnp.zeros_like(self._potential.value, dtype=self._dtype)

    def _update_states(self, current: CurrentArray) -> None:
        """
            Update neuron's soma states variables.
        """
        self._potential.value += self.potential_scale * (
            -self._potential.value + 
            (1 / self.resistance) * self.spike_slope * jnp.exp((self._potential.value - self.threshold)/self.spike_slope) +
            self.resistance * current.value
        )

    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        # Compute spikes.
        spikes = jnp.greater(self._potential.value, self.threshold.value).astype(self._dtype)
        # Reset neurons.
        self._potential.value = spikes * self.potential_reset + (1 - spikes) * self._potential.value
        return SpikeArray(spikes)
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################


@register_config
class RefractoryExponentialSomaConfig(ExponentialSomaConfig):
    cooldown: float = dc.field(
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
            units: Shape
            potential_rest: float
            potential_reset: float
            potential_tau: float
            resistance: float
            threshold: float
            cooldown: float

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
    config: RefractoryExponentialSomaConfig

    # Type hints
    cooldown: Constant
    refractory: Variable

    def __init__(self, config: RefractoryExponentialSomaConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    # NOTE: potential_rest is substracted to potential related terms to rebase potential at zero.
    def build(self, input_specs: dict[str, InputSpec]):
        super().build(input_specs)
        # Refractory period.
        self.cooldown = Constant(self.config.cooldown, dtype=self._dtype)
        self.refractory = Variable(jnp.array(self.cooldown), dtype=self._dtype)

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
        is_ready = jnp.greater(self.refractory.value, self.cooldown).astype(self._dtype)
        self._potential.value += self.potential_scale * (-self._potential.value + is_ready * self.resistance * current.value)

    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        # Compute spikes.
        spikes = jnp.logical_and(jnp.greater(self._potential.value, self.threshold.value), 
                                 jnp.greater(self.refractory.value, self.cooldown)).astype(self._dtype)
        # Reset neurons.
        self._potential.value = spikes * self.potential_reset + (1 - spikes) * self._potential.value
        # Set neuron refractory period.
        self.refractory.value = (1 - spikes) * (self.refractory.value + self._dt)
        return SpikeArray(spikes)
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class AdaptiveExponentialSomaConfig(RefractoryExponentialSomaConfig):
    threshold_tau: float = dc.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Adaptive action potential threshold decay constant.',
        })
    threshold_delta: float = dc.field(
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
class AdaptiveExponentialSoma(RefractoryExponentialSoma):
    """
        Adaptive Exponential soma model.

        Init:
            units: Shape
            potential_rest: float
            potential_reset: float
            potential_tau: float
            resistance: float
            threshold: float
            cooldown: float
            threshold_tau: float
            threshold_delta: float

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
    config: AdaptiveExponentialSomaConfig

    # Type hints
    threshold: Tracer

    def __init__(self, config: AdaptiveExponentialSomaConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    # NOTE: potential_rest is substracted to potential related terms to rebase potential at zero.
    def build(self, input_specs: dict[str, InputSpec]):
        super().build(input_specs)
        # Overwrite constant threshold with a tracer.
        self.threshold = Tracer(
            self._shape,
            tau=self.config.threshold_tau, 
            base=(self.config.threshold - self.config.potential_rest), 
            scale=self.config.threshold_delta, 
            dt=self._dt, dtype=self._dtype
        )

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