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

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class IzhikevichSomaConfig(SomaConfig):
    """
        IzhikevichSoma model configuration class.
    """

    potential_rest: float | jax.Array = dc.field(
        default = -65.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Membrane rest potential.',
        })
    potential_reset: float | jax.Array = dc.field(
        default = -65.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Membrane after spike reset potential. ' +
                           'C parameter for the Izhikevich model.',
        })
    resistance: float | jax.Array = dc.field(
        default = 100.0,
        metadata = {
            'units': 'MΩ', # [1/µS]
            'validators': [
                TypeValidator,
            ], 
            'description': 'Membrane resistance.',
        })
    threshold: float | jax.Array = dc.field(
        default = 30.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Action potential threshold base value.',
        })
    recovery_timescale: float | jax.Array = dc.field(
        default = 0.02, 
        metadata = {
            'units': '',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Time scale of the recovery variable. ' +
                           'A parameter for the Izhikevich model.',
        })
    recovery_sensitivity: float | jax.Array = dc.field(
        default = 0.2, 
        metadata = {
            'units': '',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Sensitivity of the recovery variable to the subthreshold fluctuations of the membrane potential. ' +
                           'B parameter for the Izhikevich model.',
        })
    recovery_update: float | jax.Array = dc.field(
        default = 2, 
        metadata = {
            'units': '',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Recovery increment after spike. ' +
                           'D parameter for the Izhikevich model.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class IzhikevichSoma(Soma):
    """
        Izhikevich soma model.

        Init:
            units: Shape
            potential_rest: float | jax.Array
            potential_reset: float | jax.Array
            resistance: float | jax.Array
            threshold: float | jax.Array
            recovery_timescale: float | jax.Array
            recovery_sensitivity: float | jax.Array
            recovery_update: float | jax.Array

        Input:
            in_spikes: SpikeArray
            
        Output:
            out_spikes: SpikeArray

        Reference: 
            Simple Model of Spiking Neurons
            Eugene M. Izhikevich
            IEEE Transactions on Neural Networks, vol. 14, no. 6, pp. 1569-1572, Nov. 2003
            https://doi.org/10.1109/TNN.2003.820440
    """
    config: IzhikevichSomaConfig

    def __init__(self, config: IzhikevichSomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    # NOTE: potential_rest is substracted to potential related terms to rebase potential at zero.
    def build(self, input_specs: dict[str, InputSpec]) -> None:
        super().build(input_specs)
        # Initialize variables.
        # Membrane. Substract potential_rest to potential related terms to rebase potential at zero.
        self.potential_reset = Constant(self.config.potential_reset - self.config.potential_rest, dtype=self._dtype)
        # Recovery
        self.recovery = Variable(jnp.zeros(self._shape, dtype=self._dtype), dtype=self._dtype)
        self.recovery_update = Constant(self.config.recovery_update, dtype=self._dtype)
        self.recovery_timescale = Constant(self.config.recovery_timescale, dtype=self._dtype)
        self.recovery_sensitivity = Constant(self.config.recovery_sensitivity, dtype=self._dtype)
        # Conductance.
        self.resistance = Constant(self.config.resistance / 1000.0, dtype=self._dtype) # Current is in pA for stability
        # Threshold.
        self.threshold = Constant(self.config.threshold - self.config.potential_rest, dtype=self._dtype)

    def reset(self) -> None:
        """
            Resets component state.
        """
        super().reset()
        self.recovery.value = jnp.zeros(self._shape, dtype=self._dtype)

    def _update_states(self, current: CurrentArray) -> None:
        """
            Update neuron's soma states variables.
        """
        potential_delta = 0.04 * (self._potential.value - self.config.potential_rest)**2 + \
            5 * (self._potential.value - self.config.potential_rest) + \
            140 - self.resistance * self.recovery.value + self.resistance * current.value
        self._potential.value += self._dt * potential_delta
        recovery_delta = self.recovery_timescale * (
            self.recovery_sensitivity * (self._potential.value - self.config.potential_rest) - self.recovery.value)
        self.recovery.value += self._dt * recovery_delta

    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        # Compute spikes.
        spikes = jnp.greater(self._potential.value, self.threshold.value).astype(self._dtype)
        # Reset neurons.
        self._potential.value = spikes * self.potential_reset + (1 - spikes) * self._potential.value
        # Update recovery.
        self.recovery.value = spikes * (self.recovery.value + self.recovery_update) + (1 - spikes) * self.recovery.value
        return SpikeArray(spikes)
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################