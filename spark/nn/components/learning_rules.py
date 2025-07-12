#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax.numpy as jnp
from dataclasses import dataclass
from typing import TypedDict, Dict, List
from spark.core.tracers import Tracer
from spark.core.payloads import SparkPayload, SpikeArray, FloatArray
from spark.core.variable_containers import ConfigDict
from spark.core.shape import bShape, Shape, normalize_shape
from spark.core.registry import register_module
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
        #'cd':1.0,           # [ms]
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
        self._params = params

    @property
    @abc.abstractmethod
    def input_shapes(self,) -> List[Shape]:
        pass

    @property
    @abc.abstractmethod
    def output_shapes(self,) -> List[Shape]:
        pass

    @abc.abstractmethod
    def _compute_kernel_update(self, *args: SparkPayload) -> FloatArray:
        """
            Computes next kernel update.
        """
        pass

    @abc.abstractmethod
    def __call__(self, *args: SparkPayload) -> LearningRuleOutput:
        """
            Computes and returns the next kernel update.
        """
        pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class HebbianRule(LearningRule):
    """
        Hebbian plasticy rule model.
    """

    def __init__(self, 
                 input_shape: bShape,
                 output_shape: bShape,
                 params: Dict,
                 async_spikes: bool,
                 **kwargs):
        super().__init__(params=params, **kwargs)
        # Initialize shapes
        self._input_shape = normalize_shape(input_shape)
        self._output_shape = normalize_shape(output_shape)
        self._kernel_shape = self._input_shape
        self._async_spikes = async_spikes
        # Set missing parameters to default values.
        for k in Hebbian_cfgs.DEFAULT:
            self._params.setdefault(k,  Hebbian_cfgs.DEFAULT[k])
        self._params = ConfigDict(config=self._params)
        # Tracers.
        self.pre_trace = Tracer(self._kernel_shape, tau=self._params['pre_tau'], scale=1/self._params['pre_tau'], dtype=self._dtype) 
        self.post_trace = Tracer(self._output_shape, tau=self._params['post_tau'], scale=1/self._params['post_tau'], dtype=self._dtype)
        self.post_slow_trace = Tracer(self._output_shape, tau=self._params['post_slow_tau'], scale=1/self._params['post_slow_tau'], dtype=self._dtype)
        self.target_trace = Tracer(self._kernel_shape, tau=self._params['target_tau'], scale=1/self._params['target_tau'], dtype=self._dtype)
        # Tranpose axis.
        if len(self._output_shape) == 1 and len(self._output_shape) == 1:
            self._ax_transpose = [1, 0]
        else:
            self._ax_transpose = list(range(len(self._output_shape), len(self._kernel_shape))) +\
                                list(range(len(self._output_shape)))

    @property
    def input_shapes(self,) -> List[Shape]:
        return [self._input_shape, self._output_shape, self._kernel_shape]
    
    @property
    def output_shapes(self,) -> List[Shape]:
        return [self._kernel_shape]

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
        delta_target = current_kernel - self._params['P'] * target_trace * (1/4 - target_trace) * (1/2 - target_trace)
        target_trace = self.target_trace(delta_target)
        # Transpose
        if self._async_spikes:
            pre_spikes = jnp.transpose(pre_spikes, self._ax_transpose)
            pre_trace = jnp.transpose(pre_trace, self._ax_transpose)
            current_kernel = jnp.transpose(current_kernel, self._ax_transpose)
            target_trace = jnp.transpose(target_trace, self._ax_transpose)
            # Update
            dK = self._params['gamma'] * (self._params['a'] * pre_trace * post_slow_trace * post_spikes +                       # Triplet LTP
                                          self._params['b'] * post_trace * pre_spikes +                                         # Doublet LTD
                                          self._params['c'] * (current_kernel - target_trace) * (post_trace**3) * post_spikes + # Heterosynaptic plasticity.
                                          self._params['d'] * pre_spikes)                                                       # Transmitter induced.
            kernel = current_kernel + self._dt * dK
            return jnp.transpose(kernel, self._ax_transpose)
        else: 
            pre_spikes = pre_spikes.reshape(1, -1)
            post_spikes = post_spikes.reshape(-1, 1)
            post_slow_trace = post_slow_trace.reshape(-1, 1)
            post_trace = post_trace.reshape(-1, 1)
            # Update
            dK = self._params['gamma'] * (self._params['a'] * pre_trace * post_slow_trace * post_spikes +                       # Triplet LTP
                                          self._params['b'] * post_trace * pre_spikes +                                         # Doublet LTD
                                          self._params['c'] * (current_kernel - target_trace) * (post_trace**3) * post_spikes + # Heterosynaptic plasticity.
                                          self._params['d'] * pre_spikes)                                                       # Transmitter induced.
            kernel = current_kernel + self._dt * dK
            return kernel
        
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