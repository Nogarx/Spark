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
from spark.nn.neurons import Neuron, NeuronOutput
from spark.core.payloads import SpikeArray
from spark.core.variables import Constant
from spark.nn.components import ALIFSoma, SimpleSynapses, N2NDelays, HebbianRule, DummyDelays
from spark.core.shape import bShape
from spark.core.registry import register_module

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_module
class ALIFNeuron(Neuron):
    """
        Leaky integrate and fire neuronal model.
    """

    soma: Soma
    delays: Delays
    synapses: Synanpses
    learning_rule: LearningRule

    def __init__(self, 
                 units: bShape,
                 max_delay: int = 16,
                 inhibitory_rate: float = 0.2,
                 async_spikes = True,
                 soma_params: dict = {},
                 synapses_params: dict = {},
                 learning_rule_params: dict = {},
                 **kwargs):
        super().__init__(units, **kwargs)
        # Sanity checks
        if not isinstance(inhibitory_rate, float) and inhibitory_rate >= 0 and inhibitory_rate <= 1.0:
            raise ValueError(f'"inhibitory_rate" must be a float in the range [0,1], got {inhibitory_rate}.')
        # Main attributes
        self.max_delay = max_delay
        self.inhibitory_rate = inhibitory_rate
        self.async_spikes = async_spikes
        # Temporary storage
        self._soma_params = soma_params
        self._synapses_params = synapses_params
        self._learning_rule_params = learning_rule_params
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
        self.soma = ALIFSoma(params=self._soma_params)
        del self._soma_params
        # Delays model.
        if self.async_spikes:
            self.delays = N2NDelays(output_shape=self.units, 
                                    max_delay=self.max_delay)
        else:
            self.delays = DummyDelays()
        # Synaptic model.
        self.synapses = SimpleSynapses(output_shape=self.units, 
                                    params=self._synapses_params, 
                                    async_spikes=True)
        del self._synapses_params
        # Learning rule model.
        self.learning_rule = HebbianRule(output_shape=self.units, 
                                         params=self._learning_rule_params,
                                         async_spikes=True)
        del self._learning_rule_params

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