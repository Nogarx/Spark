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
class QuadrupletRuleConfig(LearningRuleConfig):
    """
    QuadrupletRule configuration class.
    """

    pre_tau: float | jax.Array = dc.field(
        default = 10.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the  presynaptic spike train',
        })
    post_tau: float | jax.Array = dc.field(
        default = 10.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the postsynaptic spike train',
        })
    q_alpha: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Alpha parameter of the quadruplet rule',
        })
    q_beta: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Beta parameter of the quadruplet rule',
        })
    q_gamma: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Gamma parameter of the quadruplet rule',
        })
    q_delta: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Delta parameter of the quadruplet rule',
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
class QuadrupletRule(LearningRule):
    """
        Quadruplet plasticy rule model.

        Init:
            pre_tau: float | jax.Array
            post_tau: float | jax.Array
            q_alpha: float | jax.Array
            q_beta: float | jax.Array
            q_gamma: float | jax.Array
            q_delta: float | jax.Array
            eta: float | jax.Array

        Input:
            modulation: FloatArray
            pre_spikes: SpikeArray
            post_spikes: SpikeArray
            kernel: FloatArray
            
        Output:
            kernel: FloatArray
    """
    config: QuadrupletRuleConfig

    def __init__(self, config: QuadrupletRuleConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, input_specs: dict[str, PortSpecs]):
        # Initialize shapes
        input_shape = input_specs['pre_spikes'].shape
        output_shape = input_specs['post_spikes'].shape
        kernel_shape = input_specs['kernel'].shape
        ker_dtype = input_specs['kernel'].dtype
        # Initialize variables.
        _pre_tau = self.config.pre_tau.init(key=self.get_rng_keys(1), shape=kernel_shape, dtype=ker_dtype)
        _post_tau = self.config.post_tau.init(key=self.get_rng_keys(1), shape=kernel_shape, dtype=ker_dtype)
        # Tracers.
        self.pre_trace = Tracer(kernel_shape, tau=_pre_tau, scale=1/_pre_tau, dtype=ker_dtype, dt=self._dt) 
        self.post_trace = Tracer(output_shape, tau=_post_tau, scale=1/_post_tau, dtype=ker_dtype, dt=self._dt)
        self.eta = Constant(self.config.eta, dtype=ker_dtype)
        self.q_alpha = Constant(self.config.q_alpha, dtype=ker_dtype)
        self.q_beta = Constant(self.config.q_beta, dtype=ker_dtype)
        self.q_gamma = Constant(self.config.q_gamma, dtype=ker_dtype)
        self.q_delta = Constant(self.config.q_delta, dtype=ker_dtype)
        # Einsum labels.
        async_spikes = input_specs['pre_spikes'].async_spikes
        self._post_pre_dot = get_einsum_dot_exp_string(output_shape, input_shape, side='left' if async_spikes else 'none')

    def reset(self) -> None:
        """
            Resets component state.
        """
        self.pre_trace.reset()
        self.post_trace.reset()

    def _compute_kernel_update(self, modulation: FloatArray, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> jax.Array:
        """
            Computes next kernel update.
        """
        # Update and get current trace value
        _pre_spikes = pre_spikes.spikes.squeeze()
        _post_spikes = post_spikes.spikes.squeeze()
        _kernel = kernel.value.squeeze()
        pre_trace = self.pre_trace(_pre_spikes)
        post_trace = self.post_trace(_post_spikes)
        # Compute rule
        dK = self.eta * (modulation.value + 0.1) * (
            + self.q_alpha * _pre_spikes 
            + self.q_beta * jnp.einsum(self._post_pre_dot, post_trace, _pre_spikes)
            + self.q_gamma * _post_spikes
            + self.q_delta * jnp.einsum(self._post_pre_dot, _post_spikes, pre_trace)
        )
        new_kernel = jnp.clip(_kernel + self._dt * dK, min=0.0)
        return new_kernel
        
    def __call__(
            self, 
            modulation: FloatArray, 
            pre_spikes: SpikeArray, 
            post_spikes: SpikeArray, 
            kernel: FloatArray
        ) -> LearningRuleOutput:
        """
            Computes and returns the next kernel update.
        """
        return {
            'kernel': FloatArray(self._compute_kernel_update(modulation, pre_spikes, post_spikes, kernel))
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class QuadrupletRuleTensorConfig(LearningRuleConfig):
    """
    QuadrupletRuleTensor configuration class.
    """

    pre_tau: tuple[float, float, float, float] = dc.field(
        default = (10.0, 10.0, 10.0, 10.0), 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the presynaptic spike train (order: EE,EI,IE,II).',
        })
    post_tau: tuple[float, float, float, float] = dc.field(
        default = (10.0, 10.0, 10.0, 10.0), 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the postsynaptic spike train (order: EE,EI,IE,II).',
        })
    q_alpha: tuple[float, float, float, float] = dc.field(
        default = (1.0, 1.0, 1.0, 1.0), 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Alpha parameter of the quadruplet rule (order: EE,EI,IE,II).',
        })
    q_beta: tuple[float, float, float, float] = dc.field(
        default = (1.0, 1.0, 1.0, 1.0), 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Beta parameter of the quadruplet rule (order: EE,EI,IE,II).',
        })
    q_gamma: tuple[float, float, float, float] = dc.field(
        default = (1.0, 1.0, 1.0, 1.0), 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Gamma parameter of the quadruplet rule (order: EE,EI,IE,II).',
        })
    q_delta: tuple[float, float, float, float] = dc.field(
        default = (1.0, 1.0, 1.0, 1.0), 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Delta parameter of the quadruplet rule (order: EE,EI,IE,II).',
        })
    max_clip: tuple[float, float, float, float] = dc.field(
        default = (20.0, 20.0, 20.0, 20.0), 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Maximum allowed synaptic weight (order: EE,EI,IE,II).',
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
class QuadrupletRuleTensor(LearningRule):
    """
        Quadruplet plasticy rule model (tensor).

        Init:
            pre_tau: float | jax.Array
            post_tau: float | jax.Array
            eta: float | jax.Array

        Input:
            modulation: FloatArray
            pre_spikes: SpikeArray
            post_spikes: SpikeArray
            kernel: FloatArray
            
        Output:
            kernel: FloatArray
    """
    config: QuadrupletRuleTensorConfig

    def __init__(self, config: QuadrupletRuleTensorConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, input_specs: dict[str, PortSpecs]):
        # Initialize shapes
        input_shape = input_specs['pre_spikes'].shape
        output_shape = input_specs['post_spikes'].shape
        kernel_shape = input_specs['kernel'].shape
        ker_dtype = input_specs['kernel'].dtype
        # Synapses types
        post_inhibition_mask = input_specs['post_spikes'].inhibition_mask
        if input_specs['pre_spikes'].async_spikes: 
            pre_inhibition_mask = input_specs['pre_spikes'].inhibition_mask[0]
        else:
            pre_inhibition_mask = input_specs['pre_spikes'].inhibition_mask 
        synapses_type_mask = (
            + 0*jnp.outer(1-post_inhibition_mask, 1-pre_inhibition_mask) # EE
            + 1*jnp.outer(1-post_inhibition_mask, pre_inhibition_mask) # EI
            + 2*jnp.outer(post_inhibition_mask, 1-pre_inhibition_mask) # IE
            + 3*jnp.outer(post_inhibition_mask, pre_inhibition_mask) # II
        )
        # Initialize variables.
        _pre_tau = jnp.zeros_like(synapses_type_mask, dtype=jnp.float16)
        _post_tau = jnp.zeros_like(synapses_type_mask, dtype=jnp.float16)
        _q_alpha = jnp.zeros_like(synapses_type_mask, dtype=jnp.float16)
        _q_beta = jnp.zeros_like(synapses_type_mask, dtype=jnp.float16)
        _q_gamma = jnp.zeros_like(synapses_type_mask, dtype=jnp.float16)
        _q_delta = jnp.zeros_like(synapses_type_mask, dtype=jnp.float16)
        _max_clip = jnp.zeros_like(synapses_type_mask, dtype=jnp.float16)
        for i in range(4):
            _pre_tau = jnp.where(synapses_type_mask == i, self.config.pre_tau[i], _pre_tau)
            _post_tau = jnp.where(synapses_type_mask == i, self.config.post_tau[i], _post_tau)
            _q_alpha = jnp.where(synapses_type_mask == i, self.config.q_alpha[i], _q_alpha)
            _q_beta = jnp.where(synapses_type_mask == i, self.config.q_beta[i], _q_beta)
            _q_gamma = jnp.where(synapses_type_mask == i, self.config.q_gamma[i], _q_gamma)
            _q_delta = jnp.where(synapses_type_mask == i, self.config.q_delta[i], _q_delta)
            _max_clip = jnp.where(synapses_type_mask == i, self.config.max_clip[i], _max_clip)
        # Tracers.
        self.pre_trace = Tracer(kernel_shape, tau=_pre_tau, scale=1/_pre_tau, dtype=ker_dtype, dt=self._dt) 
        self.post_trace = Tracer(kernel_shape, tau=_post_tau, scale=1/_post_tau, dtype=ker_dtype, dt=self._dt)
        self.eta = Constant(self.config.eta, dtype=ker_dtype)
        self.q_alpha = Constant(_q_alpha, dtype=ker_dtype)
        self.q_beta = Constant(_q_beta, dtype=ker_dtype)
        self.q_gamma = Constant(_q_gamma, dtype=ker_dtype)
        self.q_delta = Constant(_q_delta, dtype=ker_dtype)
        self.max_clip = Constant(_max_clip, dtype=ker_dtype)
        self.synapses_type_mask = Constant(synapses_type_mask, dtype=jnp.uint8)
        # Einsum labels.
        self._ker_pre_dot = get_einsum_dot_exp_string(kernel_shape, input_shape, side='right')
        self._ker_post_dot = get_einsum_dot_exp_string(kernel_shape, output_shape, side='left')
        self._post_reshape = output_shape + (len(kernel_shape) - len(output_shape)) * (1,)

    def reset(self) -> None:
        """
            Resets component state.
        """
        self.pre_trace.reset()
        self.post_trace.reset()

    def _compute_kernel_update(self, modulation: FloatArray, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> jax.Array:
        """
            Computes next kernel update.
        """
        # Update and get current trace value
        _pre_spikes = pre_spikes.spikes.squeeze()
        _post_spikes = post_spikes.spikes.squeeze()
        _kernel = kernel.value.squeeze()
        pre_trace = self.pre_trace(_pre_spikes)
        post_trace = self.post_trace(_post_spikes.reshape(self._post_reshape))
        # Compute rule
        dK = self.eta * modulation.value * (
            + self.q_alpha * _pre_spikes 
            + self.q_beta * jnp.einsum(self._ker_pre_dot, post_trace, _pre_spikes)
            + self.q_gamma * _post_spikes.reshape(self._post_reshape)
            + self.q_delta * jnp.einsum(self._ker_post_dot, pre_trace, _post_spikes)
        )
        new_kernel = jnp.clip(_kernel + self._dt * dK, min=0.0, max=self.max_clip.value)
        return new_kernel
        
    def __call__(
            self, 
            modulation: FloatArray, 
            pre_spikes: SpikeArray, 
            post_spikes: SpikeArray, 
            kernel: FloatArray
        ) -> LearningRuleOutput:
        """
            Computes and returns the next kernel update.
        """
        return {
            'kernel': FloatArray(self._compute_kernel_update(modulation, pre_spikes, post_spikes, kernel))
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################