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
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator
from spark.nn.initializers.common import NormalizedSparseUniformInitializerConfig
from spark.nn.components.synapses.base import Synanpses, SynanpsesConfig
from spark.nn.initializers.base import Initializer, InitializerConfig

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
    kernel_initializer: InitializerConfig = dc.field(
        default_factory = NormalizedSparseUniformInitializerConfig,
        metadata = {
            'units': 'pA',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Synaptic weights initializer method. Note that we require the kernel entries to be in pA for numerical stability.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class LinearSynapses(Synanpses):
    """
        Linea synaptic model. 
        Output currents are computed as the dot product of the kernel with the input spikes.

        Init:
            units: tuple[int, ...]
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
        

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize shapes
        self.async_spikes = input_specs['spikes'].async_spikes
        self.async_spikes = self.async_spikes if self.async_spikes is not None else False
        self._input_shape = utils.validate_shape(input_specs['spikes'].shape)
        self._real_input_shape = self._input_shape[len(self._output_shape):] if self.async_spikes else self._input_shape
        self._sum_axes = tuple(range(len(self._output_shape), len(self._output_shape)+len(self._real_input_shape)))
        # Get kernel initializer
        initializer_cls: type[Initializer] = self.config.kernel_initializer.class_ref
        # Override initializer config
        initializer = initializer_cls(
            config=self.config.kernel_initializer, 
            norm_axes = tuple(s for s in range(len(self._output_shape))),
        )
        # Initialize kernel
        kernel = initializer(key=self.get_rng_keys(1), shape=self._output_shape+self._real_input_shape)
        self.kernel = Variable(kernel, dtype=self._dtype)
        
    def get_kernel(self,) -> FloatArray:
        return FloatArray(self.kernel.value)

    def get_flat_kernel(self,) -> FloatArray:
        return FloatArray(self.kernel.value.reshape(prod(self._output_shape), prod(self._real_input_shape)))

    def set_kernel(self, new_kernel: FloatArray) -> None:
        self.kernel.value = new_kernel.value

    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        return CurrentArray(jnp.sum(self.kernel.value * spikes.value, axis=self._sum_axes) * 1000.0)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################