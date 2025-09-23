#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import jax.numpy as jnp
import dataclasses
from spark.core.tracers import Tracer
from spark.core.payloads import SpikeArray, CurrentArray
from spark.core.variables import Variable, Constant
from spark.core.registry import register_module
from spark.core.config import SparkConfig
from spark.core.config_validation import TypeValidator, PositiveValidator
from .base import Soma

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ALIFSomaConfig(SparkConfig):
    potential_rest: float = dataclasses.field(
        default = -60.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Membrane rest potential.',
        })
    potential_reset: float = dataclasses.field(
        default = -50.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Membrane after spike reset potential.',
        })
    potential_tau: float = dataclasses.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Membrane potential decay constant.',
        })
    resistance: float = dataclasses.field(
        default = 100.0,
        metadata = {
            'units': 'MΩ', # [1/µS]
            'validators': [
                TypeValidator,
            ], 
            'description': 'Membrane resistance.',
        })
    threshold: float = dataclasses.field(
        default = -40.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Adaptive action potential threshold base value.',
        })
    threshold_tau: float = dataclasses.field(
        default = 100.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Adaptive action potential threshold decay constant.',
        })
    threshold_delta: float = dataclasses.field(
        default = 100.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Adaptive action potential threshold after spike increment.',
        })
    cooldown: float = dataclasses.field(
        default = 2.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Soma refractory period.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class LIFSomaConfig(ALIFSomaConfig):
    threshold_tau: float = dataclasses.field(
        default = 1.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Adaptive action potential threshold decay constant.',
        })
    threshold_delta: float = dataclasses.field(
        default = 0.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Adaptive action potential threshold after spike increment.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ALIFSoma(Soma):
    """
        ALIF soma model.
    """
    config: ALIFSomaConfig

    def __init__(self, config: ALIFSomaConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, input_specs: dict[str, InputSpec]):
        super().build(input_specs)
        # Initialize variables.
        # Membrane. Substract potential_rest to potential related terms to rebase potential at zero.
        self._potential_reset = Constant(self.config.potential_reset - self.config.potential_rest, dtype=self._dtype)
        self._potential_scale = Constant(self._dt / self.config.potential_tau, dtype=self._dtype)
        # Conductance.
        self._resistance = Constant(self.config.resistance / 1000.0, dtype=self._dtype) # Current is in pA for stability
        # Threshold.
        self._threshold = Tracer(self._shape,
                                 tau=self.config.threshold_tau, 
                                 base=(self.config.threshold - self.config.potential_rest), 
                                 scale=self.config.threshold_delta, 
                                 dt=self._dt, dtype=self._dtype)
        # Refractory period.
        self.cooldown = Constant(self.config.cooldown, dtype=self._dtype)
        self.refractory = Variable(jnp.array(self.cooldown), dtype=self._dtype)

    def reset(self) -> None:
        """
            Resets component state.
        """
        self._potential.value = jnp.zeros_like(self._potential.value, dtype=self._dtype)
        self.refractory.value = jnp.array(self.cooldown, dtype=self._dtype)
        self._threshold.reset()

    def _update_states(self, I: CurrentArray) -> None:
        """
            Update neuron's soma states variables.
        """
        is_ready = jnp.greater(self.refractory.value, self.cooldown).astype(self._dtype)
        self._potential.value += self._potential_scale * (-self._potential.value + is_ready * self._resistance * I.value)

    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        # Compute spikes.
        spikes = jnp.logical_and(jnp.greater(self._potential.value, self._threshold.value), 
                                 jnp.greater(self.refractory.value, self.cooldown)).astype(self._dtype)
        # Reset neurons.
        self._potential.value = spikes * self._potential_reset + (1 - spikes) * self._potential.value
        # Update thresholds
        self._threshold(spikes)
        # Set neuron refractory period.
        self.refractory.value = (1 - spikes) * (self.refractory.value + self._dt)
        return SpikeArray(spikes)
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################