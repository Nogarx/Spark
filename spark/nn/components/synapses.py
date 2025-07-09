#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax.numpy as jnp
from math import prod
from typing import TypedDict, Dict, List
from dataclasses import dataclass
from spark.nn.components.base import Component
from spark.core.tracers import Tracer, DoubleTracer
from spark.core.shape import bShape, Shape, normalize_shape
from spark.core.payloads import SpikeArray, CurrentArray, FloatArray, BooleanMask
from spark.core.variable_containers import Variable, ConfigDict
from spark.core.registry import register_module
from spark.nn.initializers.kernel import KernelInitializer, sparse_uniform_kernel_initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@dataclass
class SimpleConfigurations:
    TRACED = {
        'tau': 5.0,         # [ms]
        'scale': 1.0,
    }
    DOUBLE_TRACED = {
        'tau_1': 5.0,       # [ms]
        'tau_2': 5.0,       # [ms]
        'scale_1': 1.0,
        'scale_2': 1.0,            
    }
Simple_cfgs = SimpleConfigurations()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Generic Soma output contract.
class SynanpsesOutput(TypedDict):
    currents: CurrentArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Synanpses(Component):
    """
        Abstract synapse model.
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

@register_module
class SimpleSynapses(Synanpses):
    """
        Simple synaptic model. 
        Output currents are computed as the dot product of the kernel with the input spikes.
    """

    def __init__(self, 
                 input_shape: bShape,
                 output_shape: bShape,
                 async_spikes: bool, 
                 params: Dict = {},
                 kernel_initializer: KernelInitializer =
                    lambda density, dtype, scale: sparse_uniform_kernel_initializer(prob=density, scale=scale, dtype=dtype),
                 **kwargs):
        # Initialize super.
        super().__init__(**kwargs)
        # Initialize shapes
        self._input_shape = normalize_shape(input_shape)
        self._output_shape = normalize_shape(output_shape)
        self._real_input_shape = self._input_shape[len(self._output_shape):] if async_spikes else self._input_shape
        self._sum_axes = tuple(range(len(self._output_shape), len(self._output_shape)+len(self._real_input_shape)))
        # Main attributes
        self._async_spikes = async_spikes
        self._kernel_initializer = kernel_initializer
        self._params = params
        # Inhibitory mask
        #self._inhibition_mask = Constant(inhibition_mask if inhibition_mask else jnp.zeros(self._input_shape), dtype=jnp.bool_)
        #self._inhibition = Constant(1 - 2 * self._inhibition_mask, dtype=self._dtype)
        # Initialize state variables
        kernel = kernel_initializer(density=self._params['kernel_density'] if 'kernel_density' in self._params else 0.2, 
                                    scale=self._params['kernel_scale'] if 'kernel_scale' in self._params else 1, 
                                    dtype=self._dtype)\
                                   (key=self.get_rng_keys(1), 
                                    input_shape=self._real_input_shape,
                                    output_shape=self._output_shape)
        self.kernel = Variable(kernel, dtype=self._dtype)


    @property
    def input_shapes(self,) -> List[Shape]:
        return [self._input_shape]

    @property
    def output_shapes(self,) -> List[Shape]:
        return [self._output_shape]

    def reset(self) -> None:
        """
            Resets component state.
        """
        return

    def get_kernel(self,) -> FloatArray:
        return FloatArray(self.kernel.value)

    def get_flat_kernel(self,) -> FloatArray:
        return FloatArray(self.kernel.value.reshape(prod(self._output_shape), prod(self._real_input_shape)))

    def set_kernel(self, new_kernel: FloatArray) -> FloatArray:
        self.kernel.value = new_kernel

    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        return CurrentArray(jnp.sum(self.kernel.value * spikes.value, axis=self._sum_axes))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class TracedSynapses(SimpleSynapses):
    """
        Traced synaptic model. 
        Output currents are computed as the trace of the dot product of the kernel with the input spikes.
    """

    def __init__(self, 
                 input_shape: bShape,
                 output_shape: bShape,
                 async_spikes: bool, 
                 double_traced: bool = False,
                 params: Dict = {},
                 inhibition_mask: BooleanMask = None,
                 kernel_initializer: KernelInitializer =
                    lambda density, dtype, scale: sparse_uniform_kernel_initializer(prob=density, scale=scale, dtype=dtype),
                 **kwargs):
        # Initialize super.
        super().__init__(input_shape, output_shape, async_spikes, params, kernel_initializer, **kwargs)
        # Internal variables.
        self._double_traced = double_traced
        # Set missing parameters to default values.
        for k in Simple_cfgs.DOUBLE_TRACED if self._double_traced else Simple_cfgs.TRACED:
            self._params.setdefault(k, Simple_cfgs.DOUBLE_TRACED[k] if self._double_traced else Simple_cfgs.TRACED[k])
        self._params = ConfigDict(config=self._params)
        # Internal variables.
        if self._double_traced:
            self.current_tracer = DoubleTracer(shape=self.kernel.value.shape,
                                               tau_1=self._params['tau_1'],
                                               tau_2=self._params['tau_2'],
                                               scale_1=self._params['scale_1']/self._params['tau_1'],
                                               scale_2=self._params['scale_2']/self._params['tau_2'],
                                               dt=self._dt,
                                               dtype=self._dtype)
        else:
            self.current_tracer = Tracer(shape=self.kernel.value.shape,
                                         tau=self._params['tau'],
                                         scale=self._params['scale']/self._params['tau'],
                                         dt=self._dt,
                                         dtype=self._dtype)

    def reset(self) -> None:
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