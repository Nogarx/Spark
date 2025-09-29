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
from spark.nn.neurons import Neuron, NeuronOutput
from spark.core.payloads import SpikeArray
from spark.core.variables import Constant
from spark.core.shape import Shape
from spark.core.registry import register_module
from spark.core.config import SparkConfig
from spark.core.config_validation import TypeValidator, PositiveValidator, ZeroOneValidator

from spark.nn.components.delays.dummy import DummyDelays
from spark.nn.components.somas.adaptive_leaky import AdaptiveLeakySoma, AdaptiveLeakySomaConfig
from spark.nn.components.delays.base import Delays
from spark.nn.components.delays.n2n_delays import N2NDelays, N2NDelaysConfig
from spark.nn.components.synapses.linear import LineaSynapses, LineaSynapsesConfig
from spark.nn.components.learning_rules.base import LearningRule, LearningRuleConfig
from spark.nn.components.learning_rules.zenke_rule import ZenkeRule, ZenkeRuleConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ALIFNeuronConfig(SparkConfig):
    units: Shape = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Shape of the pool of neurons.',
        })
    max_delay: float = dc.field(
        default = 0.2, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': '',
        })
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
    async_spikes: bool = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Use asynchronous spikes. This parameter should be True if the incomming spikes are \
                            intercepted by a delay component and False otherwise.',
        })
    soma_config: AdaptiveLeakySomaConfig = dc.field(
        metadata = {
            'description': 'Soma configuration.',
        })
    synapses_config: LineaSynapsesConfig = dc.field(
        default_factory = LineaSynapsesConfig,
        metadata = {
            'description': 'Synapses configuration.',
        })
    delays_config: N2NDelaysConfig = dc.field(
        default_factory = N2NDelaysConfig,
        metadata = {
            'description': 'Delays configuration.',
        })
    learning_rule_config: LearningRule = dc.field(
        default_factory = ZenkeRuleConfig,
        metadata = {
            'description': 'Learning configuration.',
        })
    #def __post_init__(self):
    #    field_map = {f.name: f for f in dc.fields(self)}
    #    self.synapses_config = field_map['synapses_config'].type()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ALIFNeuron(Neuron):
    """
        Leaky integrate and fire neuronal model.
    """
    config: ALIFNeuronConfig

    # Auxiliary type hints
    soma: AdaptiveLeakySoma
    delays: N2NDelays
    synapses: LineaSynapses
    learning_rule: LearningRule

    def __init__(self, config: ALIFNeuronConfig = None, **kwargs):
        super().__init__(config=config, **kwargs)
        # Main attributes
        self.max_delay = self.config.max_delay
        self.inhibitory_rate = self.config.inhibitory_rate
        self.async_spikes = self.config.async_spikes
        # Set output shapes earlier to allow cycles.
        self.set_recurrent_shape_contract(shape=self.units)

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize inhibitory mask.
        inhibitory_units = int(self._units * self.inhibitory_rate)
        indices = jax.random.permutation(self.get_rng_keys(1), jnp.arange(self._units), independent=True)[:inhibitory_units]
        inhibition_mask = jnp.ones((self._units,), dtype=self._dtype)
        inhibition_mask = inhibition_mask.at[indices].set(-1).reshape(self.units)
        self._inhibition_mask = Constant(inhibition_mask, dtype=jnp.float16)
        # Soma model.
        self.soma = AdaptiveLeakySoma(config=self.config.soma_config)
        # Delays model.
        self.delays = N2NDelays(config=self.config.delays_config) if self.async_spikes else DummyDelays()
        # Synaptic model.
        self.synapses = LineaSynapses(config=self.config.synapses_config)
        # Learning rule model.
        self.learning_rule = self.config.class_ref(config=self.config.learning_rule_config)

    def __call__(self, in_spikes: SpikeArray) -> NeuronOutput:
        """
            Update neuron's states and compute spikes.
        """
        # Inference
        delays_output = self.delays(in_spikes)
        synapses_output = self.synapses(delays_output['out_spikes'])
        soma_output = self.soma(synapses_output['currents'])
        # Learning
        learning_rule_output = self.learning_rule(delays_output['out_spikes'], soma_output['spikes'], self.synapses.get_kernel())
        #self.synapses.set_kernel(learning_rule_output['kernel'])
        self.synapses.kernel.value = learning_rule_output['kernel'].value
        # Signed spikes
        return {
            'out_spikes': SpikeArray(soma_output['spikes'].value * self._inhibition_mask)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################