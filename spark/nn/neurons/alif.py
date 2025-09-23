#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import jax
import jax.numpy as jnp
import dataclasses
from spark.nn.neurons import Neuron, NeuronOutput
from spark.core.payloads import SpikeArray
from spark.core.variables import Constant
from spark.core.shape import Shape
from spark.core.registry import register_module
from spark.core.config import SparkConfig
from spark.core.config_validation import TypeValidator, PositiveValidator, ZeroOneValidator

from spark.nn.components.delays.dummy import DummyDelays
from spark.nn.components.somas.alif import ALIFSoma, ALIFSomaConfig
from spark.nn.components.delays.n2n_delays import N2NDelays, N2NDelaysConfig
from spark.nn.components.synapses.simple import SimpleSynapses, SimpleSynapsesConfig
from spark.nn.components.learning_rules.zenke_hebian_rule import HebbianRule, HebbianLearningConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ALIFNeuronConfig(SparkConfig):
    units: Shape = dataclasses.field(
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Shape of the pool of neurons.',
        })
    max_delay: float = dataclasses.field(
        default = 0.2, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': '',
        })
    inhibitory_rate: float = dataclasses.field(
        default = 0.2, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                ZeroOneValidator,
            ],
            'description': '',
        })
    async_spikes: bool = dataclasses.field(
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Use asynchronous spikes. This parameter should be True if the incomming spikes are \
                            intercepted by a delay component and False otherwise.',
        })
    soma_config: ALIFSomaConfig = dataclasses.field(
        metadata = {
            'description': 'Soma configuration.',
        })
    synapses_config: SimpleSynapsesConfig = dataclasses.field(
        metadata = {
            'description': 'Synapses configuration.',
        })
    delays_config: N2NDelaysConfig = dataclasses.field(
        metadata = {
            'description': 'Delays configuration.',
        })
    learning_rule_config: HebbianLearningConfig = dataclasses.field(
        metadata = {
            'description': 'Learning configuration.',
        })
    #def __post_init__(self):
    #    field_map = {f.name: f for f in dataclasses.fields(self)}
    #    self.synapses_config = field_map['synapses_config'].type()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ALIFNeuron(Neuron):
    """
        Leaky integrate and fire neuronal model.
    """
    config: ALIFNeuronConfig

    # Auxiliary type hints
    soma: ALIFSoma
    delays: N2NDelays
    synapses: SimpleSynapses
    learning_rule: HebbianRule

    def __init__(self, config: ALIFNeuronConfig = None, **kwargs):
        super().__init__(config=config, **kwargs)
        # Main attributes
        self.max_delay = self.config.max_delay
        self.inhibitory_rate = self.config.inhibitory_rate
        self.async_spikes = self.config.async_spikes
        # Set output shapes earlier to allow cycles.
        self.set_output_shapes(self.units)

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize inhibitory mask.
        inhibitory_units = int(self._units * self.inhibitory_rate)
        indices = jax.random.permutation(self.get_rng_keys(1), jnp.arange(self._units), independent=True)[:inhibitory_units]
        inhibition_mask = jnp.ones((self._units,), dtype=self._dtype)
        inhibition_mask = inhibition_mask.at[indices].set(-1).reshape(self.units)
        self._inhibition_mask = Constant(inhibition_mask, dtype=jnp.float16)
        # Soma model.
        self.soma = ALIFSoma(config=self.config.soma_config)
        # Delays model.
        self.delays = N2NDelays(config=self.config.delays_config) if self.async_spikes else DummyDelays()
        # Synaptic model.
        self.synapses = SimpleSynapses(config=self.config.synapses_config)
        # Learning rule model.
        self.learning_rule = HebbianRule(config=self.config.learning_rule_config)

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