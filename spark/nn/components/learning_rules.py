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
from typing import TypedDict, Dict
from spark.core.tracers import Tracer
from spark.core.payloads import SparkPayload, SpikeArray, FloatArray
from spark.core.variables import ConfigDict, Constant
from spark.core.shape import bShape, Shape, normalize_shape
from spark.core.registry import register_module
from spark.core.utils import get_einsum_labels
from spark.core.config import SparkConfig
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.base import Component

# Oja rule
#dK = gamma * (pre_trace * post_trace - current_kernel * (post_trace**2) )  

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Generic LearningRule output contract.
class LearningRuleOutput(TypedDict):
    kernel: FloatArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class LearningRule(Component):
    """
        Abstract learning rule model.
    """

    def __init__(self, **kwargs):
        # Initialize super.
        super().__init__(**kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class HebbianLearningConfig(SparkConfig):
    async_spikes: bool = dataclasses.field(
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Use asynchronous spikes. This parameter should be True if the incomming spikes are \
                            intercepted by a delay component and False otherwise.',
        })
    pre_tau: float = dataclasses.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': '',
        })
    post_tau: float = dataclasses.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': '',
        })
    post_slow_tau: float = dataclasses.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': '',
        })
    target_tau: float = dataclasses.field(
        default = 20000.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': '',
        })
    a: float = dataclasses.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': '',
        })
    b: float = dataclasses.field(
        default = -1.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': '',
        })
    c: float = dataclasses.field(
        default = -1.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': '',
        })
    d: float = dataclasses.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': '',
        })
    p: float = dataclasses.field(
        default = 20.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': '',
        })
    gamma: float = dataclasses.field(
        default = 0.1, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': '',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class HebbianRule(LearningRule):
    """
        Hebbian plasticy rule model.

        Init:
            async_spikes: bool
            pre_tau: float
            post_tau: float
            post_slow_tau: float
            target_tau: float
            a: float
            b: float
            c: float
            d: float
            P: float
            gamma: float

        Input:
            pre_spikes: SpikeArray
            post_spikes: SpikeArray
            current_kernel: FloatArray
            
        Output:
            kernel: FloatArray
    """
    config: HebbianLearningConfig

    def __init__(self, config: HebbianLearningConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize varibles
        self._async_spikes = self.config.async_spikes

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize shapes
        input_shape = input_specs['pre_spikes'].shape
        output_shape = input_specs['post_spikes'].shape
        kernel_shape = input_specs['current_kernel'].shape
        # Tracers.
        self.pre_trace = Tracer(kernel_shape, tau=self.config.pre_tau, scale=1/self.config.pre_tau, dtype=self._dtype) 
        self.post_trace = Tracer(output_shape, tau=self.config.post_tau, scale=1/self.config.post_tau, dtype=self._dtype)
        self.post_slow_trace = Tracer(output_shape, tau=self.config.post_slow_tau, scale=1/self.config.post_slow_tau, dtype=self._dtype)
        self.target_trace = Tracer(kernel_shape, tau=self.config.target_tau, scale=1/self.config.target_tau, dtype=self._dtype)
        self.a = Constant(self.config.a)
        self.b = Constant(self.config.b)
        self.c = Constant(self.config.c)
        self.d = Constant(self.config.d)
        self.p = Constant(self.config.p)
        self.gamma = Constant(self.config.gamma)
        # Einsum labels.
        out_labels = get_einsum_labels(len(output_shape))
        in_labels = get_einsum_labels(len(input_shape), len(output_shape))
        ker_labels = get_einsum_labels(len(kernel_shape))
        self._post_pre_prod = f'{out_labels},{ker_labels if self._async_spikes else in_labels}->{ker_labels}'
        self._ker_post_prod = f'{ker_labels},{out_labels}->{ker_labels}' 
            
    def reset(self) -> None:
        """
            Resets component state.
        """
        self.pre_trace.reset()
        self.post_trace.reset()
        self.post_slow_trace.reset()
        self.target_trace.reset()

    def _compute_kernel_update(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, current_kernel: FloatArray) -> FloatArray:
        """
            Computes next kernel update.
        """
        # Unpack values to simplify code
        pre_spikes = pre_spikes.value
        post_spikes = post_spikes.value
        current_kernel = current_kernel.value
        # Update and get current trace value
        pre_trace = self.pre_trace(pre_spikes)
        post_trace = self.post_trace(post_spikes)
        post_slow_trace = self.post_slow_trace(post_spikes)
        target_trace = self.target_trace.value
        delta_target = current_kernel - self.config.p * target_trace * (1/4 - target_trace) * (1/2 - target_trace)
        target_trace = self.target_trace(delta_target)
        # Triplet LTP
        a = self.a * jnp.einsum(self._ker_post_prod, pre_trace, post_slow_trace * post_spikes)
        # Doublet LTD
        b = self.b * jnp.einsum(self._post_pre_prod, post_trace, pre_spikes)
        # Heterosynaptic plasticity.
        c = self.c * jnp.einsum(self._ker_post_prod, current_kernel - target_trace, (post_trace**3) * post_spikes)
        # Transmitter induced.
        d = self.d * pre_spikes
        # Compute rule
        dK = self.gamma * (a + b + c + d)
        return current_kernel + self._dt * dK
        
    def __call__(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, current_kernel: FloatArray) -> LearningRuleOutput:
        """
            Computes and returns the next kernel update.
        """
        return {
            'kernel': FloatArray(self._compute_kernel_update(pre_spikes, post_spikes, current_kernel))
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################