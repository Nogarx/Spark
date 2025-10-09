#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import jax.numpy as jnp
import dataclasses as dc
from math import prod
import typing as tp
import spark.core.utils as utils
from spark.core.payloads import SpikeArray, CurrentArray, FloatArray
from spark.core.variables import Variable
from spark.core.registry import register_module, register_config, REGISTRY
from spark.core.config_validation import TypeValidator
from spark.nn.initializers.kernel import KernelInitializerConfig, SparseUniformKernelInitializerConfig
from spark.nn.components.synapses.base import Synanpses, SynanpsesConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class LinearSynapsesConfig(SynanpsesConfig):
    """
        LinearSynapses model configuration class.
    """

    units: tuple[int, ...] = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'tuple[int, ...] of the postsynaptic pool of neurons.',
        })
    async_spikes: bool = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Use asynchronous spikes. This parameter should be True if the incomming spikes are \
                            intercepted by a delay component and False otherwise.',
        })
    kernel_initializer: KernelInitializerConfig = dc.field(
        default_factory = SparseUniformKernelInitializerConfig,
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Synaptic weights initializer method.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class LinearSynapses(Synanpses):
    """
        Linea synaptic model. 
        Output currents are computed as the dot product of the kernel with the input spikes.

        Init:
            units: tuple[int, ...]
            async_spikes: bool
            kernel_initializer: KernelInitializerConfig

        Input:
            spikes: SpikeArray
            
        Output:
            currents: CurrentArray


        Reference: 
            Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. 
            Gerstner W, Kistler WM, Naud R, Paninski L. 
            Chapter 1.3 Integrate-And-Fire Models
            https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
    """
    config: LinearSynapsesConfig

    def __init__(self, config: LinearSynapses | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize shapes
        self._output_shape = utils.validate_shape(self.config.units)
        # Initialize varibles
        self.async_spikes = self.config.async_spikes
        

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize shapes
        self._input_shape = utils.validate_shape(input_specs['spikes'].shape)
        self._real_input_shape = self._input_shape[len(self._output_shape):] if self.async_spikes else self._input_shape
        self._sum_axes = tuple(range(len(self._output_shape), len(self._output_shape)+len(self._real_input_shape)))
        # Initialize varibles
        initializer: tp.Callable = REGISTRY.INITIALIZERS[self.config.kernel_initializer.name].class_ref(config=self.config.kernel_initializer)
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

    def set_kernel(self, new_kernel: FloatArray) -> None:
        self.kernel.value = new_kernel.value

    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        return CurrentArray(jnp.sum(self.kernel.value * spikes.value, axis=self._sum_axes))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################