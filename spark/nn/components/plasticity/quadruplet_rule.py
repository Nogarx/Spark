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
from spark.nn.components.plasticity.base import Plasticity, PlasticityConfig, PlasticityOutput, PlasticityParamLike
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

#PlasticityParamLike = float  | tuple[float, float, float, float] | jax.Array | MaskedInitializer

@register_config
class QuadrupletRuleConfig(PlasticityConfig):
    """
    QuadrupletRule configuration class.
    """


    pre_tau: PlasticityParamLike = dc.field(
        default = 1, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the  presynaptic spike train',
        })
    post_tau:  PlasticityParamLike  = dc.field(
        default = [1.0, 2.0, 3.0, 4.0], 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the postsynaptic spike train',
        })
    q_alpha: PlasticityParamLike = dc.field(
        default = 1.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Alpha parameter of the quadruplet rule',
        })
    q_beta: PlasticityParamLike = dc.field(
        default = 1.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Beta parameter of the quadruplet rule',
        })
    q_gamma: PlasticityParamLike = dc.field(
        default = [1.0, 2.0, 3.0, 4.0], 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Gamma parameter of the quadruplet rule',
        })
    q_delta: PlasticityParamLike = dc.field(
        default = 1.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Delta parameter of the quadruplet rule',
        })
    eta: float = dc.field(
        default = 0.1, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Learning rate',
        })
    max_clip: PlasticityParamLike = dc.field(
        default = (20.0, 20.0, 20.0, 20.0), 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Maximum allowed synaptic weight (order: EE,EI,IE,II).',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class QuadrupletRule(Plasticity):
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
        # Get shapes
        input_shape = input_specs['pre_spikes'].shape
        output_shape = input_specs['post_spikes'].shape
        kernel_shape = input_specs['kernel'].shape
        ker_dtype = input_specs['kernel'].dtype
        # Initialize variables.
        # NOTE: Since variables used for plasticity may have very different shapes and initialization patterns
        # variables should almost always be initialized using self._initialize_variable. This method automatically
        # handles the most common initialization patterns and tries to keep the variable the smallest shape possible
        # that is still compute efficient for plasticity computation (THIS APPRAOCH IS NOT ALWAYS MEMORY EFFICIENT).
        _initialize_variable_fn = lambda var: self._initialize_variable(var, shape=kernel_shape, dtype=ker_dtype)
        _pre_tau = _initialize_variable_fn(self.config.init.pre_tau)
        _post_tau = _initialize_variable_fn(self.config.init.post_tau)
        _q_alpha = _initialize_variable_fn(self.config.init.q_alpha)
        _q_beta = _initialize_variable_fn(self.config.init.q_beta)
        _q_gamma = _initialize_variable_fn(self.config.init.q_gamma)
        _q_delta = _initialize_variable_fn(self.config.init.q_delta)
        _max_clip = _initialize_variable_fn(self.config.init.max_clip)
        # Tracers.
        _post_trace_shape = output_shape if len(output_shape) > len(_post_tau) else kernel_shape
        self.pre_trace = Tracer(kernel_shape, tau=_pre_tau, scale=1/_pre_tau, dtype=ker_dtype, dt=self._dt) 
        self.post_trace = Tracer(_post_trace_shape, tau=_post_tau, scale=1/_post_tau, dtype=ker_dtype, dt=self._dt)
        self.eta = Constant(self.config.eta, dtype=ker_dtype)
        self.q_alpha = Constant(_q_alpha, dtype=ker_dtype)
        self.q_beta = Constant(_q_beta, dtype=ker_dtype)
        self.q_gamma = Constant(_q_gamma, dtype=ker_dtype)
        self.q_delta = Constant(_q_delta, dtype=ker_dtype)
        self.max_clip = Constant(_max_clip, dtype=ker_dtype)
        # Einsum labels.
        async_spikes = input_specs['pre_spikes'].async_spikes
        self._post_pre_dot = get_einsum_dot_exp_string(_post_trace_shape, input_shape, side='left' if async_spikes else 'none')
        self._pre_post_dot = get_einsum_dot_exp_string(kernel_shape, output_shape, side='left')
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
        _pre_spikes = pre_spikes.spikes
        _post_spikes = post_spikes.spikes
        _kernel = kernel.value
        pre_trace = self.pre_trace(_pre_spikes)
        post_trace = self.post_trace(_post_spikes.reshape(self._post_reshape))
        # Compute rule
        dK = self.eta * modulation.value * (
            + self.q_alpha * _pre_spikes 
            + self.q_beta * jnp.einsum(self._post_pre_dot, post_trace, _pre_spikes)
            + self.q_gamma * _post_spikes.reshape(self._post_reshape)
            + self.q_delta * jnp.einsum(self._pre_post_dot, pre_trace, _post_spikes)
        )
        new_kernel = jnp.clip(_kernel + self._dt * dK, min=0.0, max=self.max_clip.value)
        return new_kernel
        

    def __call__(
            self, 
            modulation: FloatArray, 
            pre_spikes: SpikeArray, 
            post_spikes: SpikeArray, 
            kernel: FloatArray
        ) -> PlasticityOutput:
        """
            Computes and returns the next kernel update.
        """
        return {
            'kernel': FloatArray(self._compute_kernel_update(modulation, pre_spikes, post_spikes, kernel))
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################