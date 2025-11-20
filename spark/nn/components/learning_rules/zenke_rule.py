#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import jax
import dataclasses as dc
import jax.numpy as jnp
from spark.core.tracers import Tracer
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.variables import Constant
from spark.core.registry import register_module, register_config
from spark.core.utils import get_einsum_labels
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.learning_rules.base import LearningRule, LearningRuleConfig, LearningRuleOutput
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ZenkeRuleConfig(LearningRuleConfig):
    """
       ZenkeRule configuration class.
    """

    pre_tau: float | jax.Array | Initializer = dc.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the presynaptic spike train',
        })
    post_tau: float | jax.Array | Initializer = dc.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the fast postynaptic spike train',
        })
    post_slow_tau: float | jax.Array | Initializer = dc.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the slow postynaptic spike train',
        })
    target_tau: float | jax.Array | Initializer = dc.field(
        default = 20000.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the kernel target',
        })
    a: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Scale factor of the triplet LTP term.',
        })
    b: float | jax.Array = dc.field(
        default = -1.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Scale factor of the doublet LTD term.',
        })
    c: float | jax.Array = dc.field(
        default = -1.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Scale factor of the heterosynaptic plasticity term.',
        })
    d: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Scale factor of the transmitter induced plasticity term.',
        })
    p: float | jax.Array = dc.field(
        default = 20.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': '',
        })
    gamma: float | jax.Array = dc.field(
        default = 0.1, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Learning rate',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ZenkeRule(LearningRule):
    """
        Zenke plasticy rule model. This model is an extension of the classic Hebbian Rule.

        Init:
            pre_tau: float | jax.Array
            post_tau: float | jax.Array
            post_slow_tau: float | jax.Array
            target_tau: float | jax.Array
            a: float | jax.Array
            b: float | jax.Array
            c: float | jax.Array
            d: float | jax.Array
            P: float | jax.Array
            gamma: float | jax.Array

        Input:
            pre_spikes: SpikeArray
            post_spikes: SpikeArray
            current_kernel: FloatArray
            
        Output:
            kernel: FloatArray
    """
    config: ZenkeRuleConfig

    def __init__(self, config: ZenkeRuleConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, input_specs: dict[str, InputSpec]):
        self.async_spikes = input_specs['pre_spikes'].async_spikes
        self.async_spikes = self.async_spikes if self.async_spikes is not None else False
        # Initialize shapes
        input_shape = input_specs['pre_spikes'].shape
        output_shape = input_specs['post_spikes'].shape
        kernel_shape = input_specs['current_kernel'].shape
        # Initialize variables.
        _pre_tau = self.config.pre_tau.init(key=self.get_rng_keys(1), shape=kernel_shape, dtype=self._dtype)
        _post_tau = self.config.post_tau.init(key=self.get_rng_keys(1), shape=output_shape, dtype=self._dtype)
        _post_slow_tau = self.config.post_slow_tau.init(key=self.get_rng_keys(1), shape=output_shape, dtype=self._dtype)
        _target_tau = self.config.target_tau.init(key=self.get_rng_keys(1), shape=kernel_shape, dtype=self._dtype)
        # Tracers.
        self.pre_trace = Tracer(kernel_shape, tau=_pre_tau, scale=1/_pre_tau, dtype=self._dtype) 
        self.post_trace = Tracer(output_shape, tau=_post_tau, scale=1/_post_tau, dtype=self._dtype)
        self.post_slow_trace = Tracer(output_shape, tau=_post_slow_tau, scale=1/_post_slow_tau, dtype=self._dtype)
        self.target_trace = Tracer(kernel_shape, tau=_target_tau, scale=1/_target_tau, dtype=self._dtype)
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
        self._post_pre_prod = f'{out_labels},{ker_labels if self.async_spikes else in_labels}->{ker_labels}'
        self._ker_post_prod = f'{ker_labels},{out_labels}->{ker_labels}' 
            
    def reset(self) -> None:
        """
            Resets component state.
        """
        self.pre_trace.reset()
        self.post_trace.reset()
        self.post_slow_trace.reset()
        self.target_trace.reset()

    def _compute_kernel_update(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, current_kernel: FloatArray) -> jax.Array:
        """
            Computes next kernel update.
        """
        # Update and get current trace value
        pre_trace = self.pre_trace(pre_spikes.value)
        post_trace = self.post_trace(post_spikes.value)
        post_slow_trace = self.post_slow_trace(post_spikes.value)
        target_trace = self.target_trace.value
        delta_target = current_kernel.value - self.config.p * target_trace * (1/4 - target_trace) * (1/2 - target_trace)
        target_trace = self.target_trace(delta_target)
        # Triplet LTP
        a = self.a * jnp.einsum(self._ker_post_prod, pre_trace, post_slow_trace * post_spikes.value)
        # Doublet LTD
        b = self.b * jnp.einsum(self._post_pre_prod, post_trace, pre_spikes.value)
        # Heterosynaptic plasticity.
        c = self.c * jnp.einsum(self._ker_post_prod, current_kernel.value - target_trace, (post_trace**3) * post_spikes.value)
        # Transmitter induced.
        d = self.d * pre_spikes.value
        # Compute rule
        dK = self.gamma * (a + b + c + d)
        return current_kernel.value + self._dt * dK
        
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