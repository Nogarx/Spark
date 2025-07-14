#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import abc
import jax.numpy as jnp
from dataclasses import dataclass
from typing import TypedDict, Dict
from spark.core.tracers import Tracer
from spark.core.payloads import SparkPayload, SpikeArray, FloatArray
from spark.core.variables import ConfigDict
from spark.core.shape import bShape, Shape, normalize_shape
from spark.core.registry import register_module
from spark.core.utils import get_einsum_labels
from spark.nn.components.base import Component

# Oja rule
#dK = gamma * (pre_trace * post_trace - current_kernel * (post_trace**2) )  

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@dataclass
class HebbianConfigurations:
    DEFAULT = {
        'pre_tau':20.0,         # [ms]
        'post_tau':20.0,        # [ms]
        'post_slow_tau':100.0,  # [ms]
        'target_tau':20000.0,   # [ms]
        'a':1.0,            
        'b':-1.0,           
        'c':-1.0,           
        'd':1.0, 
        'P':20.0,            
        'gamma':0.1
    }
Hebbian_cfgs = HebbianConfigurations()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Generic LearningRule output contract.
class LearningRuleOutput(TypedDict):
    kernel: FloatArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class LearningRule(Component):
    """
        Abstract learning rule model.
    """

    def __init__(self, 
                 params: Dict,
                 **kwargs):
        # Initialize super.
        super().__init__(**kwargs)
        # Main attributes
        self.params = ConfigDict(params)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class HebbianRule(LearningRule):
    """
        Hebbian plasticy rule model.
    """

    def __init__(self, 
                 async_spikes: bool,
                 params: Dict = {},
                 **kwargs):
        # Initialize super.
        super().__init__(params, **kwargs)
        # Initialize varibles
        self._async_spikes = async_spikes
        # Set missing parameters to default values.
        for k in Hebbian_cfgs.DEFAULT:
            self.params.setdefault(k,  Hebbian_cfgs.DEFAULT[k])

    def build(self, input_specs: dict[str, InputSpec]):
        # Initialize shapes
        input_shape = input_specs['pre_spikes'].shape
        output_shape = input_specs['post_spikes'].shape
        kernel_shape = input_specs['current_kernel'].shape
        # Tracers.
        self.pre_trace = Tracer(kernel_shape, tau=self.params['pre_tau'], scale=1/self.params['pre_tau'], dtype=self._dtype) 
        self.post_trace = Tracer(output_shape, tau=self.params['post_tau'], scale=1/self.params['post_tau'], dtype=self._dtype)
        self.post_slow_trace = Tracer(output_shape, tau=self.params['post_slow_tau'], scale=1/self.params['post_slow_tau'], dtype=self._dtype)
        self.target_trace = Tracer(kernel_shape, tau=self.params['target_tau'], scale=1/self.params['target_tau'], dtype=self._dtype)
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
        delta_target = current_kernel - self.params['P'] * target_trace * (1/4 - target_trace) * (1/2 - target_trace)
        target_trace = self.target_trace(delta_target)
        # Triplet LTP
        a = self.params['a'] * jnp.einsum(self._ker_post_prod, pre_trace, post_slow_trace * post_spikes)
        # Doublet LTD
        b = self.params['b'] * jnp.einsum(self._post_pre_prod, post_trace, pre_spikes)
        # Heterosynaptic plasticity.
        c = self.params['c'] * jnp.einsum(self._ker_post_prod, current_kernel - target_trace, (post_trace**3) * post_spikes)
        # Transmitter induced.
        d = self.params['d'] * pre_spikes
        # Compute rule
        dK = self.params['gamma'] * (a + b + c + d)
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