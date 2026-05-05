#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs

import jax
import dataclasses as dc
import jax.numpy as jnp
from spark.core.tracers import Tracer
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.variables import Constant
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.plasticity.base import Plasticity, PlasticityConfig, PlasticityOutput
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ZenkeRuleConfig(PlasticityConfig):
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
    eta: float | jax.Array = dc.field(
        default = 0.1, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Learning rate',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ZenkeRule(Plasticity):
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
            eta: float | jax.Array

        Input:
            pre_spikes: SpikeArray
            post_spikes: SpikeArray
            kernel: FloatArray
            
        Output:
            kernel: FloatArray
    """
    config: ZenkeRuleConfig

    def __init__(self, config: ZenkeRuleConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> None:
        # Initialize variables.
        # NOTE: Since variables used for plasticity may have very different shapes and initialization patterns
        # variables should almost always be initialized using self._initialize_variable. This method automatically
        # handles the most common initialization patterns and tries to keep the variable the smallest shape possible
        # that is still compute efficient for plasticity computation (THIS APPRAOCH IS NOT ALWAYS MEMORY EFFICIENT).
        kernel_shape = kernel.shape
        pre_shape = pre_spikes.shape
        post_shape = post_spikes.shape
        _initialize_variable_fn = lambda var: self._initialize_variable(var, shape=kernel_shape, dtype=kernel.dtype)
        _pre_tau = _initialize_variable_fn(self.config.init.pre_tau)
        _post_tau = _initialize_variable_fn(self.config.init.post_tau)
        _post_slow_tau = _initialize_variable_fn(self.config.init.post_slow_tau)
        _target_tau = _initialize_variable_fn(self.config.init.target_tau)
        # Broadcast shapes
        # NOTE: Most learning rules may be reexpressed as sums of general dot products; the most robuts way to 
        # implement this is by reshaping pre and post spikes to the shape of the kernel. It also makes the math more clear.
        self._pre_shape = pre_shape if pre_shape == kernel_shape else (1,) * (len(kernel_shape) - len(pre_shape)) + pre_shape
        self._post_shape = post_shape if post_shape == kernel_shape else post_shape + (1,) * (len(kernel_shape) - len(post_shape))
        # Tracers.
        self.pre_trace = Tracer(self._pre_shape, tau=_pre_tau, scale=1/_pre_tau, dtype=self._dtype, dt=self._dt) 
        self.post_trace = Tracer(self._post_shape, tau=_post_tau, scale=1/_post_tau, dtype=self._dtype, dt=self._dt)
        self.post_slow_trace = Tracer(self._post_shape, tau=_post_slow_tau, scale=1/_post_slow_tau, dtype=self._dtype, dt=self._dt)
        self.target_trace = Tracer(kernel_shape, tau=_target_tau, scale=1/_target_tau, dtype=self._dtype, dt=self._dt)
        self.a = Constant(self.config.a)
        self.b = Constant(self.config.b)
        self.c = Constant(self.config.c)
        self.d = Constant(self.config.d)
        self.p = Constant(self.config.p)
        self.eta = Constant(self.config.eta)
            
    def reset(self) -> None:
        """
            Resets component state.
        """
        self.pre_trace.reset()
        self.post_trace.reset()
        self.post_slow_trace.reset()
        self.target_trace.reset()

    def _compute_kernel_update(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> jax.Array:
        """
            Computes next kernel update.
        """
        # Extract and reshape inputs
        _pre_spikes = pre_spikes.spikes.reshape(self._pre_shape)
        _post_spikes = post_spikes.spikes.reshape(self._post_shape)
        _kernel = kernel.value
        # Update and get current trace value
        pre_trace = self.pre_trace(_pre_spikes)
        post_trace = self.post_trace(_post_spikes)
        post_slow_trace = self.post_slow_trace(_post_spikes)
        target_trace = self.target_trace.value
        delta_target = _kernel - self.config.p * target_trace * (1/4 - target_trace) * (1/2 - target_trace)
        target_trace = self.target_trace(delta_target)
        # Triplet LTP
        a = self.a * pre_trace * post_slow_trace * _post_spikes
        # Doublet LTD
        b = self.b * post_trace * _pre_spikes
        # Heterosynaptic plasticity.
        c = self.c * (_kernel - target_trace) * (post_trace**3) * _post_spikes
        # Transmitter induced.
        d = self.d * _pre_spikes
        # Compute rule
        dK = self.eta * (a + b + c + d)
        return jnp.clip(_kernel + self._dt * dK, min=0.0)
        
    def __call__(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> PlasticityOutput:
        """
            Computes and returns the next kernel update.
        """
        return {
            'kernel': FloatArray(self._compute_kernel_update(pre_spikes, post_spikes, kernel))
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################