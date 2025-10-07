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
from math import prod
from spark.core.tracers import SaturableTracer, SaturableDoubleTracer
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.variables import Variable
from spark.core.registry import register_module
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.output.base import OutputInterface, OutputInterfaceConfig, OutputInterfaceOutput

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ExponentialIntegratorConfig(OutputInterfaceConfig):
    """
        ExponentialIntegrator configuration class.
    """

    num_outputs: int = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Number of output signals. Input spikes are distributed equally among the output signals. \
                            If num_outputs does not exactly divide the number of incomming spikes then an approximately \
                            even assigment is used.',
        })
    saturation_freq: float = dc.field(
        default = 50.0, 
        metadata = {
            'units': 'Hz',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Approximate average firing frequency at which the population needs to fire to sature the integrator.',
        })
    tau: float = dc.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Decay time constant of the membrane potential of the units of the spiker.',
        })
    shuffle: bool = dc.field(
        default = True, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Shuffles the input spikes, otherwise they are used sequentially to create the output signal.',
        })
    smooth_trace: bool = dc.field(
        default = True, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Smooths the output signal with an using a double exponential moving average instead of a single EMA.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ExponentialIntegrator(OutputInterface):
    """
        Transforms a discrete spike signal to a continuous signal.
        This transformation assumes a very simple integration model model without any type of adaptation or plasticity.
        Spikes are grouped into k non-overlaping clusters and every neuron contributes the same amount to the ouput.

        Init:
            num_outputs: int
            saturation_freq: float [Hz]
            tau: float [ms]
            shuffle: bool
            smooth_trace: bool
            
        Input:
            spikes: SpikeArray
            
        Output:
            signal: FloatArray
    """
    config: ExponentialIntegratorConfig

    def __init__(self, config: ExponentialIntegratorConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize internal variables
        self.num_outputs = self.config.num_outputs
        self.saturation_freq = self.config.saturation_freq
        self.tau = self.config.tau
        self.shuffle = self.config.shuffle
        self.smooth_trace = self.config.smooth_trace

    def build(self, input_specs: dict[str, InputSpec]) -> None:
        # Output mapping.
        in_dim = prod(input_specs['spikes'].shape)
        out_dim = self.num_outputs
        base = in_dim // out_dim
        remainder = in_dim % out_dim
        counts = jnp.concatenate([
            jnp.full(remainder, base + 1, dtype=jnp.int32),
            jnp.full(out_dim - remainder, base, dtype=jnp.int32)
        ])
        output_map = jnp.repeat(jnp.arange(out_dim, dtype=jnp.int32), counts)
        if not self.shuffle:
            output_map = output_map
        else: 
            output_map = jax.random.permutation(self.get_rng_keys(1), output_map)
        self._indices = Variable(output_map, dtype=jnp.uint32)

        # Initialize tracer
        if self.smooth_trace:
            self.trace = SaturableDoubleTracer(
                shape=self.num_outputs, 
                tau_1=self.tau, 
                tau_2=10,
                scale_1=(1000/self.saturation_freq)/(self._dt*self.tau*counts), 
                scale_2=1/10,
                dt=self._dt, 
                dtype=self._dtype
            )
        else:
            self.trace = SaturableTracer(
                shape=self.num_outputs, 
                tau=self.tau, 
                scale=(1000/self.saturation_freq)/(self._dt*self.tau*counts), 
                dt=self._dt, 
                dtype=self._dtype
            )

    def reset(self,):
        """
            Reset module to its default state.
        """
        self.trace.reset()

    def __call__(self, spikes: SpikeArray) -> OutputInterfaceOutput:
        # Flat array
        x = spikes.value.reshape(-1)
        # Count spikes in each output group.
        x = jax.ops.segment_sum(
            x, 
            self._indices, 
            indices_are_sorted=(not self.shuffle), 
            num_segments=self.num_outputs
        )
        # Integrate
        output = self.trace(x)
        return {
            'signal': FloatArray(output)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################