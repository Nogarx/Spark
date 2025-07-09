#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Dict
from spark.nn.neurons import Neuron, NeuronOutput
from spark.core.payloads import SpikeArray
from spark.core.variable_containers import Constant
from spark.nn.components import ALIFSoma, SimpleSynapses, N2NDelays, HebbianRule, DummyDelays
from spark.core.shape import bShape
from spark.core.registry import register_module
from math import prod

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_module
class ALIFNeuron(Neuron):
    """
        Leaky integrate and fire neuronal model.
    """

    def __init__(self, 
                 input_shape: bShape,
                 output_shape: bShape,
                 soma_params: Dict = {},
                 synapses_params: Dict = {},
                 plasticity_params: Dict = {},
                 max_delay: int = 16,
                 inhibitory_rate: float = 0.2,
                 input_inhibition_mask: jax.Array = None,
                 **kwargs):
        super().__init__(output_shape, input_shapes=input_shape, **kwargs)
        # Sanity checks
        if not isinstance(inhibitory_rate, float) and inhibitory_rate >= 0 and inhibitory_rate <= 1.0:
            raise ValueError(f'"inhibitory_rate" must be a float in the range [0,1], got {inhibitory_rate}.')
        # Main attributes
        self._max_delay = max_delay
        self._inhibitory_rate = inhibitory_rate
        # Initialize inhibitory mask.
        inhibitory_units = int(self.units * self._inhibitory_rate)
        indices = jax.random.permutation(self.get_rng_keys(1), jnp.arange(prod(self._output_shape)), independent=True)[:inhibitory_units]
        inhibition_mask = jnp.ones((prod(self._output_shape),), dtype=self._dtype)
        inhibition_mask = inhibition_mask.at[indices].set(-1).reshape(self._output_shape)
        self._inhibition_mask = Constant(inhibition_mask, dtype=jnp.float16)
        # Soma model.
        self.soma = ALIFSoma(shape=self._output_shape, 
                             params=soma_params)
        # Delays model.
        self.delays = N2NDelays(input_shape=self._input_shapes[0], 
                                output_shape=self._output_shape, 
                                max_delay=self._max_delay)

        # Synaptic model.
        self.synapses = SimpleSynapses(input_shape=self._output_shape+self._input_shapes[0], 
                                       output_shape=self._output_shape, 
                                       params=synapses_params, 
                                       async_spikes=True)

        # Plasticity model.
        self.plasticity = HebbianRule(input_shape=self._output_shape+self._input_shapes[0], 
                                      output_shape=self._output_shape, 
                                      params=plasticity_params,
                                      async_spikes=True)


    def reset(self):
        """
            Resets neuron states to their initial values.
        """
        self.soma.reset()
        #self.synapses.reset()
        #self.plasticity.reset()

    def __call__(self, input_spikes: SpikeArray) -> NeuronOutput:
        """
            Update neuron's states and compute spikes.
        """
        # Inference
        delays_output = self.delays(input_spikes)
        synapses_output = self.synapses(delays_output['spikes'])
        soma_output = self.soma(synapses_output['currents'])
        # Learning
        plasticity_output = self.plasticity(delays_output['spikes'], soma_output['spikes'], self.synapses.get_kernel())
        #self.synapses.set_kernel(kernel)
        self.synapses.kernel.value = plasticity_output['kernel'].value
        # Signed spikes
        return {
            'spikes': SpikeArray(soma_output['spikes'].value * self._inhibition_mask)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################