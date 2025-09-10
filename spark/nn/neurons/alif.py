#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec
    from spark.nn.components.somas import Soma
    from spark.nn.components.delays import Delays
    from spark.nn.components.synapses import Synanpses
    from spark.nn.components.learning_rules import LearningRule

import jax
import jax.numpy as jnp
import dataclasses
from spark.nn.neurons import Neuron, NeuronOutput
from spark.core.payloads import SpikeArray
from spark.core.variables import Constant
from spark.core.shape import bShape, Shape
from spark.core.registry import register_module
from spark.core.configuration import SparkConfig, PositiveValidator, ZeroOneValidator
from spark.nn.components import (ALIFSoma, ALIFSomaConfig, 
                                 SimpleSynapses, SimpleSynapsesConfig,
                                 N2NDelays, N2NDelaysConfig,
                                 HebbianRule, HebbianLearningConfig,
                                 DummyDelays)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ALIFNeuronConfig(SparkConfig):
    units: Shape = dataclasses.field(
        metadata = {
            'description': 'Shape of the pool of neurons.',
        })
    max_delay: float = dataclasses.field(
        default = 0.2, 
        metadata = {
            'units': 'ms',
            'validators': [
                PositiveValidator,
            ],
            'description': '',
        })
    inhibitory_rate: float = dataclasses.field(
        default = 0.2, 
        metadata = {
            'units': 'ms',
            'validators': [
                ZeroOneValidator,
            ],
            'description': '',
        })
    async_spikes: bool = dataclasses.field(
        metadata = {
            'description': 'Use asynchronous spikes. This parameter should be True if the incomming spikes are \
                            intercepted by a delay component and False otherwise.',
        })
    soma_config: ALIFSomaConfig = dataclasses.field(
        default_factory = ALIFSomaConfig,
        metadata = {
            'description': 'Soma configuration.',
        }),
    synapses_config: SimpleSynapsesConfig = dataclasses.field(
        default_factory = SimpleSynapsesConfig,
        metadata = {
            'description': 'Synapses configuration.',
        }),
    delays_config: N2NDelaysConfig = dataclasses.field(
        default_factory = N2NDelaysConfig,
        metadata = {
            'description': 'Delays configuration.',
        }),
    learning_rule_config: HebbianLearningConfig = dataclasses.field(
        default_factory = HebbianLearningConfig,
        metadata = {
            'description': 'Learning configuration.',
        }),

    def __post_init__(self):
        field_map = {f.name: f for f in dataclasses.fields(self)}

        self.synapses_config = field_map['synapses_config'].type()

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