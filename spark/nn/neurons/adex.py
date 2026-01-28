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
from spark.nn.neurons import Neuron, NeuronConfig, NeuronOutput
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.variables import Constant
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator, ZeroOneValidator

from spark.nn.components.somas.exponential import AdaptiveExponentialSoma, AdaptiveExponentialSomaConfig
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
class AdExNeuronConfig(NeuronConfig):
    """
        AdExNeuron configuration class.
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
    soma: AdaptiveExponentialSomaConfig = dc.field(
        default_factory = AdaptiveExponentialSomaConfig,
        metadata = {
            'description': 'Soma configuration.',
        })
    synapses: SynanpsesConfig = dc.field(
        default_factory = LinearSynapsesConfig,
        metadata = {
            'description': 'Synapses configuration.',
        })
    delays: DelaysConfig | None = dc.field(
        default_factory = N2NDelaysConfig,
        metadata = {
            'description': 'Delays configuration.',
        })
    learning_rule: LearningRuleConfig | None = dc.field(
        default_factory = HebbianRuleConfig,
        metadata = {
            'description': 'Learning configuration.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class AdExNeuron(Neuron):
    """
        Leaky Integrate and Fire neuronal model.
    """
    config: AdExNeuronConfig

    # Auxiliary type hints
    soma: AdaptiveExponentialSoma
    delays: Delays
    synapses: Synanpses
    learning_rule: LearningRule

    def __init__(self, config: AdExNeuronConfig | None = None, **kwargs):
        super().__init__(config=config, **kwargs)
        # Initialize inhibitory mask.
        inhibitory_units = int(self._units * self.config.inhibitory_rate)
        indices = jax.random.permutation(self.get_rng_keys(1), jnp.arange(self._units), independent=True)[:inhibitory_units]
        inhibition_mask = jnp.zeros((self._units,), dtype=jnp.bool)
        inhibition_mask = inhibition_mask.at[indices].set(True).reshape(self.units)
        self._inhibition_mask = Constant(inhibition_mask, dtype=jnp.bool)
        # Set output shapes earlier to allow cycles.
        contract = self._get_output_specs()
        contract['out_spikes'].shape = self.units
        contract['out_spikes'].inhibition_mask = self._inhibition_mask
        self.set_contract_output_specs(contract)

    def build(self, input_specs: dict[str, PortSpecs]):
        # Soma model.
        self.soma = self.config.soma.class_ref(config=self.config.soma)
        # Delays model.
        self._delays_active = self.config.delays is not None
        if self._delays_active:
            self.delays = self.config.delays.class_ref(config=self.config.delays)
        # Synaptic model.
        self.synapses = self.config.synapses.class_ref(config=self.config.synapses)
        # Learning rule model.
        self._learning_active = self.config.learning_rule is not None
        if self._learning_active:
            self.learning_rule = self.config.learning_rule.class_ref(config=self.config.learning_rule)

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