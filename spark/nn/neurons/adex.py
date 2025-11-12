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
from spark.nn.neurons import Neuron, NeuronConfig, NeuronOutput
from spark.core.payloads import SpikeArray
from spark.core.variables import Constant
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator, ZeroOneValidator

from spark.nn.components.somas.exponential import AdaptiveExponentialSoma, AdaptiveExponentialSomaConfig
from spark.nn.components.delays.base import Delays, DelaysConfig
from spark.nn.components.delays.n2n_delays import N2NDelaysConfig
from spark.nn.components.synapses.base import Synanpses, SynanpsesConfig
from spark.nn.components.synapses.linear import LinearSynapsesConfig
from spark.nn.components.learning_rules.base import LearningRule, LearningRuleConfig
from spark.nn.components.learning_rules.hebbian_rule import HebbianRule

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
    soma_config: AdaptiveExponentialSomaConfig = dc.field(
        default_factory = AdaptiveExponentialSomaConfig,
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
        default_factory = HebbianRule,
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
        # Set output shapes earlier to allow cycles.
        self.set_recurrent_shape_contract(shape=self.units)

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize inhibitory mask.
        inhibitory_units = int(self._units * self.config.inhibitory_rate)
        indices = jax.random.permutation(self.get_rng_keys(1), jnp.arange(self._units), independent=True)[:inhibitory_units]
        inhibition_mask = jnp.ones((self._units,), dtype=self._dtype)
        inhibition_mask = inhibition_mask.at[indices].set(-1).reshape(self.units)
        self._inhibition_mask = Constant(inhibition_mask, dtype=jnp.float16)
        # Soma model.
        self.soma = self.config.soma_config.class_ref(config=self.config.soma_config)
        # Delays model.
        self._delays_active = self.config.delays_config is not None
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
            'out_spikes': SpikeArray(soma_output['spikes'].value * self._inhibition_mask)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################