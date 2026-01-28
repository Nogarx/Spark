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
from spark.core.tracers import Tracer, RDTracer, RFSTracer
from spark.core.payloads import SpikeArray, CurrentArray
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator, ZeroOneValidator
from spark.nn.components.synapses.linear import LinearSynapses, LinearSynapsesConfig
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class TracedSynapsesConfig(LinearSynapsesConfig):
    """
        TracedSynapses model configuration class.
    """

    tau: float | jax.Array | Initializer = dc.field(
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
            kernel: KernelInitializerConfig
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

    def build(self, input_specs: dict[str, PortSpecs]):
        # Initialize shapes
        super().build(input_specs)
        # Initialize variables.
        _tau = self.config.tau.init(key=self.get_rng_keys(1), shape=self.kernel.value.shape, dtype=self._dtype)
        # Current tracer.
        self.current_tracer = Tracer(
            shape=self.kernel.value.shape,
            tau=_tau,
            scale=self.config.scale,
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
class RDTracedSynapsesConfig(LinearSynapsesConfig):
    """
        RDTracedSynapses model configuration class.
    """

    tau_rise: float | jax.Array | Initializer = dc.field(
        default = 1.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Rise tracer decay constant.',
    })
    scale_rise: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Rise tracer spike scaling.',
    })
    base_rise: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Rise tracer rest value.',
    })
    tau_decay: float | jax.Array | Initializer = dc.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Decay tracer decay constant.',
    })
    scale_decay: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Decay tracer spike scaling.',
    })
    base_decay: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Decay tracer rest value.',
    })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class RDTracedSynapses(LinearSynapses):
    """
        Rise-Decay traced synaptic model. 
        Output currents are computed as the RDTrace of the dot product of the kernel with the input spikes.

        Init:
            units: tuple[int, ...]
            kernel: KernelInitializerConfig
            tau_rise: float | jax.Array
            scale_rise: float | jax.Array
            base_rise: float | jax.Array
            tau_decay: float | jax.Array
            scale_decay: float | jax.Array
            base_decay: float | jax.Array

        Input:
            spikes: SpikeArray
            
        Output:
            currents: CurrentArray
    """
    config: RDTracedSynapsesConfig

    # Auxiliary type hints
    current_tracer: RDTracer

    def __init__(self, config: RDTracedSynapsesConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, input_specs: dict[str, PortSpecs]):
        # Initialize shapes
        super().build(input_specs)
        # Initialize variables.
        _tau_rise = self.config.tau_rise.init(key=self.get_rng_keys(1), shape=self.kernel.value.shape, dtype=self._dtype)
        _tau_decay = self.config.tau_decay.init(key=self.get_rng_keys(1), shape=self.kernel.value.shape, dtype=self._dtype)
        # Current tracer.
        self.current_tracer = RDTracer(
            shape=self.kernel.value.shape,
            tau_rise=_tau_rise,
            tau_decay=_tau_decay,
            scale_rise=self.config.scale_rise,
            scale_decay=self.config.scale_decay,
            base_rise=self.config.base_rise,
            base_decay=self.config.base_decay,
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
class RFSTracedSynapsesConfig(LinearSynapsesConfig):
    """
        RFSTracedSynapses model configuration class.
    """
    alpha: float | jax.Array = dc.field(
        default = 0.8, 
        metadata = {
            'validators': [
                TypeValidator,
                ZeroOneValidator,
            ],
            'description': 'Fast-Slow blending factor.',
    })
    tau_rise: float | jax.Array | Initializer = dc.field(
        default = 1.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Rise tracer decay constant.',
    })
    scale_rise: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Rise tracer spike scaling.',
    })
    base_rise: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Rise tracer rest value.',
    })
    tau_fast_decay: float | jax.Array | Initializer = dc.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Fast-decay tracer decay constant.',
    })
    scale_fast_decay: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Fast-decay tracer spike scaling.',
    })
    base_fast_decay: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Fast-decay tracer rest value.',
    })
    tau_slow_decay: float | jax.Array | Initializer = dc.field(
        default = 50.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Slow-decay tracer decay constant.',
    })
    scale_slow_decay: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Slow-decay tracer spike scaling.',
    })
    base_slow_decay: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Slow-decay tracer rest value.',
    })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class RFSTracedSynapses(LinearSynapses):
    """
        Traced synaptic model. 
        Output currents are computed as the trace of the dot product of the kernel with the input spikes.

        Init:
            units: tuple[int, ...]
            kernel: KernelInitializerConfig
            alpha: float | jax.Array
            tau_rise: float | jax.Array
            scale_rise: float | jax.Array
            base_rise: float | jax.Array
            tau_fast_decay: float | jax.Array
            scale_fast_decay: float | jax.Array
            base_fast_decay: float | jax.Array
            tau_slow_decay: float | jax.Array
            scale_slow_decay: float | jax.Array
            base_slow_decay: float | jax.Array

        Input:
            spikes: SpikeArray
            
        Output:
            currents: CurrentArray
    """
    config: RFSTracedSynapsesConfig

    # Auxiliary type hints
    current_tracer: RDTracer

    def __init__(self, config: RFSTracedSynapsesConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, input_specs: dict[str, PortSpecs]):
        # Initialize shapes
        super().build(input_specs)
        # Initialize variables.
        _tau_rise = self.config.tau_rise.init(key=self.get_rng_keys(1), shape=self.kernel.value.shape, dtype=self._dtype)
        _tau_fast_decay = self.config.tau_fast_decay.init(key=self.get_rng_keys(1), shape=self.kernel.value.shape, dtype=self._dtype)
        _tau_slow_decay = self.config.tau_slow_decay.init(key=self.get_rng_keys(1), shape=self.kernel.value.shape, dtype=self._dtype)
        # Current tracer.
        self.current_tracer = RFSTracer(
            shape=self.kernel.value.shape,
            alpha=self.config.alpha,
            tau_rise=_tau_rise,
            tau_fast_decay=_tau_fast_decay,
            tau_slow_decay=_tau_slow_decay,
            scale_rise=self.config.scale_rise,
            scale_fast_decay=self.config.scale_fast_decay,
            scale_slow_decay=self.config.scale_slow_decay,
            base_rise=self.config.base_rise,
            base_fast_decay=self.config.base_fast_decay,
            base_slow_decay=self.config.base_slow_decay,
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