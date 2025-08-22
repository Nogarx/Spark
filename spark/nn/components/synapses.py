#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import abc
import jax.numpy as jnp
import dataclasses
from math import prod
from typing import TypedDict, Callable
from spark.nn.components.base import Component
from spark.core.tracers import Tracer, DoubleTracer
from spark.core.shape import bShape, Shape, normalize_shape
from spark.core.payloads import SpikeArray, CurrentArray, FloatArray
from spark.core.variables import Variable
from spark.core.registry import register_module, REGISTRY
from spark.core.configuration import SparkConfig, PositiveValidator
from spark.nn.initializers.kernel import KernelInitializerConfig, SparseUniformKernelInitializerConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic Soma output contract.
class SynanpsesOutput(TypedDict):
    currents: CurrentArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Synanpses(Component):
    """
        Abstract synapse model.

        Init:

        Input:
            spikes: SpikeArray
            
        Output:
            currents: CurrentArray
    """

    def __init__(self, **kwargs):
        # Initialize super.
        super().__init__(**kwargs)

    @abc.abstractmethod
    def get_kernel(self,) -> FloatArray:
        pass

    @abc.abstractmethod
    def set_kernel(self, new_kernel: FloatArray) -> FloatArray:
        pass

    @abc.abstractmethod
    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        pass

    def __call__(self, spikes: SpikeArray) -> SynanpsesOutput:
        """
            Compute synanpse's currents.
        """
        return {
            'currents': self._dot(spikes)
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SimpleSynapsesConfig(SparkConfig):
    target_units_shape: Shape = dataclasses.field(
        metadata = {
            'description': 'Shape after the merge operation.',
        })
    async_spikes: bool = dataclasses.field(
        metadata = {
            'description': 'Use asynchronous spikes. This parameter should be True if the incomming spikes are \
                            intercepted by a delay component and False otherwise.',
    })
    kernel_initializer: KernelInitializerConfig = dataclasses.field(default_factory = SparseUniformKernelInitializerConfig)
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class SimpleSynapses(Synanpses):
    """
        Simple synaptic model. 
        Output currents are computed as the dot product of the kernel with the input spikes.

        Init:
            target_units_shape: Shape
            async_spikes: bool
            kernel_initializer: KernelInitializerConfig

        Input:
            spikes: SpikeArray
            
        Output:
            currents: CurrentArray
    """
    config: SimpleSynapsesConfig

    def __init__(self, config: SimpleSynapses = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize shapes
        self._output_shape = normalize_shape(self.config.target_units_shape)
        # Initialize varibles
        self.async_spikes = self.config.async_spikes

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize shapes
        self._input_shape = normalize_shape(input_specs['spikes'].shape)
        self._real_input_shape = self._input_shape[len(self._output_shape):] if self.async_spikes else self._input_shape
        self._sum_axes = tuple(range(len(self._output_shape), len(self._output_shape)+len(self._real_input_shape)))
        # Initialize varibles
        initializer: Callable = REGISTRY.INITIALIZERS[self.config.kernel_initializer.name].class_ref(self.config.kernel_initializer)
        kernel = initializer(key=self.get_rng_keys(1), input_shape=self._real_input_shape, output_shape=self._output_shape)
        self.kernel = Variable(kernel, dtype=self._dtype)
        # Inhibitory mask
        #self._inhibition_mask = Constant(inhibition_mask if inhibition_mask else jnp.zeros(self._input_shape), dtype=jnp.bool_)
        #self._inhibition = Constant(1 - 2 * self._inhibition_mask, dtype=self._dtype)
        # Initialize state variables
        # Lock params
        
    def get_kernel(self,) -> FloatArray:
        return FloatArray(self.kernel.value)

    def get_flat_kernel(self,) -> FloatArray:
        return FloatArray(self.kernel.value.reshape(prod(self._output_shape), prod(self._real_input_shape)))

    def set_kernel(self, new_kernel: FloatArray) -> FloatArray:
        self.kernel.value = new_kernel

    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        return CurrentArray(jnp.sum(self.kernel.value * spikes.value, axis=self._sum_axes))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TracedSynapsesConfig(SimpleSynapsesConfig):
    tau: float = dataclasses.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                PositiveValidator,
            ],
            'description': 'Tracer decay constant.',
    })
    scale: float = dataclasses.field(
        default = 1.0, 
        metadata = {
            'validators': [
                PositiveValidator,
            ],
            'description': 'Tracer spike scaling.',
    })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class TracedSynapses(SimpleSynapses):
    """
        Traced synaptic model. 
        Output currents are computed as the trace of the dot product of the kernel with the input spikes.

        Init:
            target_units_shape: Shape
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
                PositiveValidator,
            ],
            'description': 'First tracer decay constant.',
    })
    scale_1: float = dataclasses.field(
        default = 1.0, 
        metadata = {
            'validators': [
                PositiveValidator,
            ],
            'description': 'First tracer spike scaling.',
    })
    tau_2: float = dataclasses.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                PositiveValidator,
            ],
            'description': 'Second tracer decay constant.',
    })
    scale_2: float = dataclasses.field(
        default = 1.0, 
        metadata = {
            'validators': [
                PositiveValidator,
            ],
            'description': 'Second tracer spike scaling.',
    })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class DoubleTracedSynapses(TracedSynapses):
    """
        Traced synaptic model. 
        Output currents are computed as the trace of the dot product of the kernel with the input spikes.

        Init:
            target_units_shape: Shape
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
            dt=self.config.dt,
            dtype=self.config.dtype
        )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################