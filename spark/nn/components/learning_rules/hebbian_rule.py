#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING

from numpy import square
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs

import jax
import dataclasses as dc
import jax.numpy as jnp
from spark.core.tracers import Tracer
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.variables import Constant
from spark.core.registry import register_module, register_config
from spark.core.utils import get_einsum_dot_exp_string
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
    eta: float = dc.field(
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
            kernel: FloatArray
            
        Output:
            kernel: FloatArray
    """
    config: HebbianRuleConfig

    def __init__(self, config: HebbianRuleConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, input_specs: dict[str, PortSpecs]):
        # Initialize shapes
        input_shape = input_specs['pre_spikes'].shape
        output_shape = input_specs['post_spikes'].shape
        kernel_shape = input_specs['kernel'].shape
        # Initialize variables.
        _pre_tau = self.config.pre_tau.init(key=self.get_rng_keys(1), shape=input_shape, dtype=self._dtype)
        _post_tau = self.config.post_tau.init(key=self.get_rng_keys(1), shape=output_shape, dtype=self._dtype)
        # Tracers.
        self.pre_trace = Tracer(input_shape, tau=_pre_tau, scale=1/_pre_tau, dtype=self._dtype, dt=self._dt) 
        self.post_trace = Tracer(output_shape, tau=_post_tau, scale=1/_post_tau, dtype=self._dtype, dt=self._dt)
        self.eta = Constant(self.config.eta)
        # Einsum labels.
        async_spikes = input_specs['pre_spikes'].async_spikes
        self._post_pre_dot = get_einsum_dot_exp_string(output_shape, input_shape, side='left' if async_spikes else 'none')
            
    def reset(self) -> None:
        """
            Resets component state.
        """
        self.pre_trace.reset()
        self.post_trace.reset()

    def _compute_kernel_update(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> jax.Array:
        """
            Computes next kernel update.
        """
        # Squeeze
        _pre_spikes = pre_spikes.spikes.squeeze()
        _post_spikes = post_spikes.spikes.squeeze()
        _kernel = kernel.value.squeeze()
        # Update and get current trace value
        pre_trace = self.pre_trace(_pre_spikes)
        post_trace = self.post_trace(_post_spikes)
        # Compute rule
        dK = self.eta * (
            + jnp.einsum(self._post_pre_dot, post_trace, _pre_spikes)
            + jnp.einsum(self._post_pre_dot, _post_spikes, pre_trace)
        )
        return jnp.clip(_kernel + self._dt * dK, min=0.0)
        
    def __call__(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> LearningRuleOutput:
        """
            Computes and returns the next kernel update.
        """
        return {
            'kernel': FloatArray(self._compute_kernel_update(pre_spikes, post_spikes, kernel))
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class OjaRuleConfig(LearningRuleConfig):
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
    eta: float = dc.field(
        default = 0.1, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Learning rate',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class OjaRule(LearningRule):
    """
        Oja's plasticy rule model.

        Init:
            pre_tau: float | jax.Array
            post_tau: float | jax.Array
            gamma: float | jax.Array

        Input:
            pre_spikes: SpikeArray
            post_spikes: SpikeArray
            kernel: FloatArray
            
        Output:
            kernel: FloatArray
    """
    config: OjaRuleConfig

    def __init__(self, config: OjaRuleConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, input_specs: dict[str, PortSpecs]):
        # Initialize shapes
        input_shape = input_specs['pre_spikes'].shape
        output_shape = input_specs['post_spikes'].shape
        kernel_shape = input_specs['kernel'].shape
        # Initialize variables.
        _post_tau = self.config.post_tau.init(key=self.get_rng_keys(1), shape=output_shape, dtype=self._dtype)
        # Tracers.
        self.post_trace = Tracer(output_shape, tau=_post_tau, scale=1/_post_tau, dtype=self._dtype, dt=self._dt)
        self.eta = Constant(self.config.eta)
        # Einsum labels.
        async_spikes = input_specs['pre_spikes'].async_spikes
        self._post_pre_dot = get_einsum_dot_exp_string(output_shape, input_shape, side='left' if async_spikes else 'none')
        self._ker_post_dot = get_einsum_dot_exp_string(kernel_shape, output_shape, side='left')

    def reset(self) -> None:
        """
            Resets component state.
        """
        self.post_trace.reset()

    def _compute_kernel_update(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> jax.Array:
        """
            Computes next kernel update.
        """
        # Squeeze
        _pre_spikes = pre_spikes.spikes.squeeze()
        _post_spikes = post_spikes.spikes.squeeze()
        _kernel = kernel.value.squeeze()
        # Update and get current trace value
        post_trace = self.post_trace(_post_spikes)
        # Compute rule
        dK = self.eta * (
            + jnp.einsum(self._post_pre_dot, post_trace, _pre_spikes)
            - jnp.einsum(self._ker_post_dot, _kernel, jnp.square(post_trace))
        )
        return jnp.clip(_kernel + self._dt * dK, min=0.0)

    def __call__(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> LearningRuleOutput:
        """
            Computes and returns the next kernel update.
        """
        return {
            'kernel': FloatArray(self._compute_kernel_update(pre_spikes, post_spikes, kernel))
        }
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################