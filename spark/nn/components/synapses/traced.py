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
from spark.core.tracers import Tracer, DoubleTracer
from spark.core.payloads import SpikeArray, CurrentArray
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.synapses.linear import LinearSynapses, LinearSynapsesConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class TracedSynapsesConfig(LinearSynapsesConfig):
    """
        TracedSynapses model configuration class.
    """

    tau: float | jax.Array = dc.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Tracer decay constant.',
    })
    scale: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Tracer spike scaling.',
    })
    base: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Tracer rest value.',
    })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class TracedSynapses(LinearSynapses):
    """
        Traced synaptic model. 
        Output currents are computed as the trace of the dot product of the kernel with the input spikes.

        Init:
            units: tuple[int, ...]
            async_spikes: bool
            kernel_initializer: KernelInitializerConfig
            tau: float | jax.Array
            scale: float | jax.Array
            base: float | jax.Array

        Input:
            spikes: SpikeArray
            
        Output:
            currents: CurrentArray
    """
    config: TracedSynapsesConfig

    # Auxiliary type hints
    current_tracer: Tracer

    def __init__(self, config: TracedSynapsesConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize shapes
        super().build(input_specs)
        # Internal variables.
        self.current_tracer = Tracer(
            shape=self.kernel.value.shape,
            tau=self.config.tau,
            scale=self.config.scale/self.config.tau,
            base=self.config.base,
            dt=self.config.dt,
            dtype=self.config.dtype
        )

    def reset(self,) -> None:
        """
            Resets component state.
        """
        self.current_tracer.reset()

    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        trace = self.current_tracer(self.kernel.value * spikes.value)
        return CurrentArray(jnp.sum(trace, axis=self._sum_axes))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class DoubleTracedSynapsesConfig(LinearSynapsesConfig):
    """
        DoubleTracedSynapses model configuration class.
    """

    tau_1: float | jax.Array = dc.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'First tracer decay constant.',
    })
    scale_1: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'First tracer spike scaling.',
    })
    base_1: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'First tracer rest value.',
    })
    tau_2: float | jax.Array = dc.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Second tracer decay constant.',
    })
    scale_2: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Second tracer spike scaling.',
    })
    base_2: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Second tracer rest value.',
    })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class DoubleTracedSynapses(LinearSynapses):
    """
        Traced synaptic model. 
        Output currents are computed as the trace of the dot product of the kernel with the input spikes.

        Init:
            units: tuple[int, ...]
            async_spikes: bool
            kernel_initializer: KernelInitializerConfig
            tau_1: float | jax.Array
            scale_1: float | jax.Array
            base_1: float | jax.Array
            tau_2: float | jax.Array
            scale_2: float | jax.Array
            base_2: float | jax.Array

        Input:
            spikes: SpikeArray
            
        Output:
            currents: CurrentArray
    """
    config: DoubleTracedSynapsesConfig

    # Auxiliary type hints
    current_tracer: DoubleTracer

    def __init__(self, config: DoubleTracedSynapsesConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize shapes
        super().build(input_specs)
        # Internal variables.
        self.current_tracer = DoubleTracer(
            shape=self.kernel.value.shape,
            tau_1=self.config.tau_1,
            tau_2=self.config.tau_2,
            scale_1=self.config.scale_1/self.config.tau_1,
            scale_2=self.config.scale_2/self.config.tau_2,
            base_1=self.config.base_1,
            base_2=self.config.base_2,
            dt=self.config.dt,
            dtype=self.config.dtype
        )

    def reset(self,) -> None:
        """
            Resets component state.
        """
        self.current_tracer.reset()

    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        trace = self.current_tracer(self.kernel.value * spikes.value)
        return CurrentArray(jnp.sum(trace, axis=self._sum_axes))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################