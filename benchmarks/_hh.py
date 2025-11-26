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
from spark.core.config_validation import TypeValidator, ZeroOneValidator
from spark.nn.components.somas.base import Soma, SomaConfig
from spark.core.registry import register_module, register_config

from spark.nn.neurons import Neuron, NeuronConfig, NeuronOutput
from spark.nn.components.delays.base import Delays, DelaysConfig
from spark.nn.components.delays.n2n_delays import N2NDelaysConfig
from spark.nn.components.synapses.base import Synanpses, SynanpsesConfig
from spark.nn.components.synapses.linear import LinearSynapsesConfig
from spark.nn.components.learning_rules.base import LearningRule, LearningRuleConfig
from spark.nn.components.learning_rules.hebbian_rule import HebbianRuleConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class HodgkinHuxleySomaConfig(SomaConfig):
    """
        HodgkinHuxleySoma model configuration class.
    """

    c_m: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'units': 'μF',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Membrane capacitance.',
        })
    e_leak: float | jax.Array = dc.field(
        default = 10.6, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Reversal potential leak.',
        })
    e_na: float | jax.Array = dc.field(
        default = 115,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Reversal potential sodium.',
        })
    e_k: float | jax.Array = dc.field(
        default = -12.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Reversal potential potassium.',
        })
    g_leak: float | jax.Array = dc.field(
        default = 0.3, 
        metadata = {
            'units': 'mS',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Conductance leak.',
        })
    g_na: float | jax.Array = dc.field(
        default = 120, 
        metadata = {
            'units': 'mS',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Conductance sodium.',
        })
    g_k: float | jax.Array = dc.field(
        default = 36, 
        metadata = {
            'units': 'mS',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Conductance potassium.',
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
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class HodgkinHuxleySoma(Soma):
    """
        Hodgkin-Huxley soma model.

        Init:
            units: tuple[int, ...]
            c_m: float | jax.Array
            e_leak: float | jax.Array
            e_na: float | jax.Array
            e_k: float | jax.Array
            g_leak: float | jax.Array
            g_na: float | jax.Array
            g_k: float | jax.Array
            threshold: float | jax.Array
    """
    config: HodgkinHuxleySomaConfig

    def __init__(self, config: HodgkinHuxleySomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    # NOTE: potential_rest is substracted to potential related terms to rebase potential at zero.
    def build(self, input_specs: dict[str, InputSpec]) -> None:
        super().build(input_specs)
        # Initialize variables.
        # Reversal potentials
        self.e_leak = Constant(self.config.e_leak, dtype=self._dtype)
        self.e_na = Constant(self.config.e_na, dtype=self._dtype)
        self.e_k = Constant(self.config.e_k, dtype=self._dtype)
        # Conductance
        self.g_leak = Constant(self.config.g_leak, dtype=self._dtype)
        self.g_na = Constant(self.config.g_na, dtype=self._dtype)
        self.g_k = Constant(self.config.g_k, dtype=self._dtype)
        # Gates
        self.m = Variable(jnp.ones(self.units, dtype=self._dtype), dtype=self._dtype)
        self.h = Variable(jnp.zeros(self.units, dtype=self._dtype), dtype=self._dtype)
        self.n = Variable(0.5*jnp.ones(self.units, dtype=self._dtype), dtype=self._dtype)
        # Membrane.
        self.c_m = Constant(self.config.c_m, dtype=self._dtype)
        self.threshold = Constant(self.config.threshold, dtype=self._dtype)
        self.potential.value = self.e_leak*jnp.ones(self.units, dtype=self._dtype)
        self.refractory = Variable(jnp.zeros(self.units, dtype=jnp.bool), dtype=jnp.bool)

    def reset(self) -> None:
        """
            Resets component state.
        """
        self.m.value = jnp.ones(self.units, dtype=self._dtype)
        self.h.value = jnp.zeros(self.units, dtype=self._dtype)
        self.n.value = 0.5*jnp.ones(self.units, dtype=self._dtype)
        self.potential.value = self.e_leak*jnp.ones(self.units, dtype=self._dtype)
        self.refractory.value = jnp.zeros(self.units, dtype=jnp.bool)

    def _update_states(self, current: CurrentArray) -> None:
        """
            Update neuron's soma states variables.
        """
        # Update potential
        g_na = self.g_na * (self.m**3) * self.h
        g_k  = self.g_k  * (self.n**4)
        g_sum = self.g_leak + g_na + g_k
        potential_delta = self._dt * g_sum / self.c_m
        V_inf = (
            self.g_leak * self.e_leak 
            + g_na * self.e_na 
            + g_k * self.e_k 
            + current
        ) / g_sum
        self.potential.value = jnp.exp(-potential_delta) * self.potential.value + (1 - jnp.exp(-potential_delta)) * V_inf
        # Update M
        alpha_m = (0.1) * (-self.potential+25) / (jnp.exp( (-self.potential+25)/(10) ) - 1)
        beta_m = 4 * jnp.exp(-self.potential/(18))
        m_delta = self._dt * (alpha_m + beta_m)
        self.m.value = jnp.exp(-m_delta) * self.m.value + (1 - jnp.exp(-m_delta)) * (alpha_m / (alpha_m + beta_m))
        # Update N
        alpha_n = (0.01) * (-self.potential+10) / (jnp.exp( (-self.potential+10)/(10) ) - 1)
        beta_n = 0.125 * jnp.exp(-self.potential/(80) )
        n_delta = self._dt * (alpha_n + beta_n)
        self.n.value = jnp.exp(-n_delta) * self.n.value + (1 - jnp.exp(-n_delta)) * (alpha_n / (alpha_n + beta_n))
        # Update H
        alpha_h = 0.07 * jnp.exp( -self.potential/(20) )
        beta_h = 1 / (jnp.exp( (-self.potential+30)/(10) ) + 1)
        h_delta = self._dt * (alpha_h + beta_h)
        self.h.value = jnp.exp(-h_delta) * self.h.value + (1 - jnp.exp(-h_delta)) * (alpha_h / (alpha_h + beta_h))
        
    

    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        # Compute spikes.
        spikes = jnp.logical_and(
            ~self.refractory.value,
            jnp.greater(self.potential.value, self.threshold.value)
        ).astype(self._dtype)
        # Refractory
        self.refractory.value = jnp.greater(self.potential.value, self.threshold.value)
        return SpikeArray(spikes)
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class HodgkinHuxleyNeuronConfig(NeuronConfig):
    """
        LIFNeuron configuration class.
    """

    inhibitory_rate: float = dc.field(
        default = 0.2, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                ZeroOneValidator,
            ],
            'description': '',
        })
    soma_config: HodgkinHuxleySomaConfig = dc.field(
        default_factory = HodgkinHuxleySomaConfig,
        metadata = {
            'description': 'Soma configuration.',
        })
    synapses_config: SynanpsesConfig = dc.field(
        default_factory = LinearSynapsesConfig,
        metadata = {
            'description': 'Synapses configuration.',
        })
    delays_config: DelaysConfig | None = dc.field(
        default_factory = N2NDelaysConfig,
        metadata = {
            'description': 'Delays configuration.',
        })
    learning_rule_config: LearningRuleConfig | None = dc.field(
        default_factory = HebbianRuleConfig,
        metadata = {
            'description': 'Learning configuration.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class HodgkinHuxleyNeuron(Neuron):
    """
        Leaky Integrate and Fire neuronal model.
    """
    config: HodgkinHuxleyNeuronConfig

    # Auxiliary type hints
    soma: HodgkinHuxleySoma
    delays: Delays
    synapses: Synanpses
    learning_rule: LearningRule

    def __init__(self, config: HodgkinHuxleyNeuronConfig | None = None, **kwargs):
        super().__init__(config=config, **kwargs)
        # Set output shapes earlier to allow cycles.
        self.set_recurrent_shape_contract(shape=self.units)

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize inhibitory mask.
        inhibitory_units = int(self._units * self.config.inhibitory_rate)
        indices = jax.random.permutation(self.get_rng_keys(1), jnp.arange(self._units), independent=True)[:inhibitory_units]
        inhibition_mask = jnp.zeros((self._units,), dtype=jnp.bool)
        inhibition_mask = inhibition_mask.at[indices].set(True).reshape(self.units)
        self._inhibition_mask = Constant(inhibition_mask, dtype=jnp.float16)
        # Soma model.
        self.soma = self.config.soma_config.class_ref(config=self.config.soma_config)
        # Delays model.
        self._delays_active = self.config.learning_rule_config is not None
        if self._delays_active:
            self.delays = self.config.delays_config.class_ref(config=self.config.delays_config)
        # Synaptic model.
        self.synapses = self.config.synapses_config.class_ref(config=self.config.synapses_config)
        # Learning rule model.
        self._learning_active = self.config.learning_rule_config is not None
        if self._learning_active:
            self.learning_rule = self.config.learning_rule_config.class_ref(config=self.config.learning_rule_config)

    def __call__(self, in_spikes: SpikeArray) -> NeuronOutput:
        """
            Update neuron's states and compute spikes.
        """
        # Inference
        if self._delays_active:
            delays_output = self.delays(in_spikes)
            synapses_output = self.synapses(delays_output['out_spikes'])
        else: 
            synapses_output = self.synapses(in_spikes)
        soma_output = self.soma(synapses_output['currents'])
        # Learning
        if self._learning_active:
            learning_rule_output = self.learning_rule(delays_output['out_spikes'], soma_output['spikes'], self.synapses.get_kernel())
            self.synapses.set_kernel(learning_rule_output['kernel'])
        # Signed spikes
        return {
            'out_spikes': SpikeArray(soma_output['spikes'].spikes, inhibition_mask=self._inhibition_mask)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################