#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import jax.numpy as jnp
import dataclasses
from spark.core.tracers import Tracer, DoubleTracer
from spark.core.payloads import SpikeArray, CurrentArray
from spark.core.registry import register_module
from spark.core.config_validation import TypeValidator, PositiveValidator
from .simple import SimpleSynapses, SimpleSynapsesConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TracedSynapsesConfig(SimpleSynapsesConfig):
    tau: float = dataclasses.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Tracer decay constant.',
    })
    scale: float = dataclasses.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Tracer spike scaling.',
    })
    base: float = dataclasses.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Tracer rest value.',
    })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class TracedSynapses(SimpleSynapses):
    """
        Traced synaptic model. 
        Output currents are computed as the trace of the dot product of the kernel with the input spikes.

        Init:
            target_units: Shape
            async_spikes: bool
            kernel_initializer: KernelInitializerConfig
            tau: float
            scale: float

        Input:
            spikes: SpikeArray
            
        Output:
            currents: CurrentArray
    """
    config: TracedSynapsesConfig

    # Auxiliary type hints
    current_tracer: Tracer

    def __init__(self, config: TracedSynapsesConfig = None, **kwargs):
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

class DoubleTracedSynapsesConfig(SimpleSynapsesConfig):
    tau_1: float = dataclasses.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'First tracer decay constant.',
    })
    scale_1: float = dataclasses.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'First tracer spike scaling.',
    })
    base_1: float = dataclasses.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'First tracer rest value.',
    })
    tau_2: float = dataclasses.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Second tracer decay constant.',
    })
    scale_2: float = dataclasses.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Second tracer spike scaling.',
    })
    base_2: float = dataclasses.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Second tracer rest value.',
    })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class DoubleTracedSynapses(SimpleSynapses):
    """
        Traced synaptic model. 
        Output currents are computed as the trace of the dot product of the kernel with the input spikes.

        Init:
            target_units: Shape
            async_spikes: bool
            kernel_initializer: KernelInitializerConfig
            tau_1: float
            scale_1: float
            tau_2: float
            scale_2: float

        Input:
            spikes: SpikeArray
            
        Output:
            currents: CurrentArray
    """
    config: DoubleTracedSynapsesConfig

    # Auxiliary type hints
    current_tracer: DoubleTracer

    def __init__(self, config: DoubleTracedSynapsesConfig = None, **kwargs):
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