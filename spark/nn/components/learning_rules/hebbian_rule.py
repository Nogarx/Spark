#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING

from numpy import square
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
class HebbianRuleConfig(LearningRuleConfig):
    """
       HebbianRule configuration class.
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
            'description': 'Time constant of the postsynaptic spike train',
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
class HebbianRule(LearningRule):
    """
        Hebbian plasticy rule model.

        Init:
            pre_tau: float | jax.Array
            post_tau: float | jax.Array
            gamma: float | jax.Array

        Input:
            pre_spikes: SpikeArray
            post_spikes: SpikeArray
            current_kernel: FloatArray
            
        Output:
            kernel: FloatArray
    """
    config: HebbianRuleConfig

    def __init__(self, config: HebbianRuleConfig | None = None, **kwargs):
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
        _post_tau = self.config.post_tau.init(key=self.get_rng_keys(1), shape=kernel_shape, dtype=self._dtype)
        # Tracers.
        self.pre_trace = Tracer(kernel_shape, tau=_pre_tau, scale=1/_pre_tau, dtype=self._dtype) 
        self.post_trace = Tracer(output_shape, tau=_post_tau, scale=1/_post_tau, dtype=self._dtype)
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

    def _compute_kernel_update(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, current_kernel: FloatArray) -> jax.Array:
        """
            Computes next kernel update.
        """
        # Update and get current trace value
        pre_trace = self.pre_trace(pre_spikes.value)
        post_trace = self.post_trace(post_spikes.value)
        # Compute rule
        dK = self.gamma * (
            jnp.einsum(self._post_pre_prod, post_trace, pre_spikes.value)
            + jnp.einsum(self._ker_post_prod, pre_trace, post_spikes.value)
        )
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

@register_config
class OjaRuleConfig(HebbianRuleConfig):
    stabilization_factor: bool = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Use asynchronous spikes. This parameter should be True if the incomming spikes are \
                            intercepted by a delay component and False otherwise.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class OjaRule(HebbianRule):
    """
        Oja's plasticy rule model.

        Init:
            pre_tau: float | jax.Array
            post_tau: float | jax.Array
            gamma: float | jax.Array

        Input:
            pre_spikes: SpikeArray
            post_spikes: SpikeArray
            current_kernel: FloatArray
            
        Output:
            kernel: FloatArray
    """

    def _compute_kernel_update(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, current_kernel: FloatArray) -> jax.Array:
        """
            Computes next kernel update.
        """
        # Update and get current trace value
        pre_trace = self.pre_trace(pre_spikes.value)
        post_trace = self.post_trace(post_spikes.value)
        # Compute rule
        dK = self.gamma * (
            jnp.einsum(self._ker_post_prod, pre_trace, post_spikes.value)
            - jnp.einsum(self._ker_post_prod, current_kernel.value, jnp.square(post_trace))
        )
        return current_kernel.value + self._dt * dK

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################