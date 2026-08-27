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
from spark.core.backend import Constant
from spark.core.registry import register_module, register_config
from spark.core.utils import get_einsum_dot_exp_string
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.plasticity.base import Plasticity, PlasticityConfig, PlasticityOutput
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class HebbianRuleConfig(PlasticityConfig):
    """
        Configuration for `HebbianRule`.

        Parameters
        ----------
        pre_tau : float or jax.Array or Initializer, default 20.0
            Decay constant of the presynaptic trace, in ms. May be a 4-tuple, one value per
            connection type.
        post_tau : float or jax.Array or Initializer, default 20.0
            Decay constant of the postsynaptic trace, in ms. May be a 4-tuple, one value per
            connection type.
        eta : float, default 0.01
            Learning rate.
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
        default = 0.01, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Learning rate',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class HebbianRule(Plasticity):
    r"""
        Pair-based Hebbian rule.

        Every presynaptic spike is potentiated in proportion to the postsynaptic trace, and every
        postsynaptic spike in proportion to the presynaptic trace. The rule only potentiates, so
        the weights grow without bound unless something else keeps them in check.

        Parameters
        ----------
        config : HebbianRuleConfig
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        pre_spikes : SpikeArray
            Presynaptic spikes, after any conduction delay.
        post_spikes : SpikeArray
            Postsynaptic spikes emitted on this step.
        kernel : FloatArray
            Current synaptic weights, read from the synapse.

        Output Ports
        ------------
        kernel : FloatArray
            The updated weights, written back onto the synapse as an effect.

        Notes
        -----
        With :math:`x` the presynaptic trace and :math:`y` the postsynaptic trace,

        .. math::
            \Delta W = \eta \left( y \, s_{\mathrm{pre}} + x \, s_{\mathrm{post}} \right)

        applied as :math:`W \leftarrow \max(W + \Delta t \, \Delta W, 0)`. Each trace is scaled by
        the reciprocal of its own time constant, so its magnitude does not change with ``tau``.

        References
        ----------
        .. [1] W. Gerstner, W. M. Kistler, R. Naud and L. Paninski, "Neuronal Dynamics: From
               Single Neurons to Networks and Models of Cognition", Chapter 19.2, Hebbian Rate
               Models. https://neuronaldynamics.epfl.ch/online/Ch19.S2.html

        See Also
        --------
        OjaRule : The same potentiation with a normalizing term.
    """
    config: HebbianRuleConfig

    def __init__(self, config: HebbianRuleConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, pre_spikes: SpikeArray,post_spikes: SpikeArray, kernel: FloatArray) -> None:
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
        # Broadcast shapes
        # NOTE: Most learning rules may be reexpressed as sums of general dot products; the most robuts way to 
        # implement this is by reshaping pre and post spikes to the shape of the kernel. It also makes the math more clear.
        self._pre_shape = pre_shape if pre_shape == kernel_shape else (1,) * (len(kernel_shape) - len(pre_shape)) + pre_shape
        self._post_shape = post_shape if post_shape == kernel_shape else post_shape + (1,) * (len(kernel_shape) - len(post_shape))
        # Tracers.
        self.pre_trace = Tracer(self._pre_shape, tau=_pre_tau, scale=1/_pre_tau, dtype=self._dtype, dt=self._dt) 
        self.post_trace = Tracer(self._post_shape, tau=_post_tau, scale=1/_post_tau, dtype=self._dtype, dt=self._dt)
        self.eta = Constant(self.config.eta)

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
        # Extract and reshape inputs
        _pre_spikes = pre_spikes.spikes.reshape(self._pre_shape)
        _post_spikes = post_spikes.spikes.reshape(self._post_shape)
        _kernel = kernel.value
        # Update and get current trace value
        pre_trace = self.pre_trace(_pre_spikes)
        post_trace = self.post_trace(_post_spikes)
        # Compute rule
        dK = self.eta.value * (
            + post_trace * _pre_spikes
            + pre_trace * _post_spikes
        )
        return jnp.clip(_kernel + self._dt * dK, min=0.0)
        
    def __call__(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> PlasticityOutput:
        """
            Computes the weights for the next step.

            Parameters
            ----------
            pre_spikes : SpikeArray
                Presynaptic spikes, after any conduction delay.
            post_spikes : SpikeArray
                Postsynaptic spikes emitted on this step.
            kernel : FloatArray
                Current synaptic weights.

            Returns
            -------
            PlasticityOutput
                Dictionary with one entry, ``kernel``, the updated weights.
        """
        return {
            'kernel': FloatArray(self._compute_kernel_update(pre_spikes, post_spikes, kernel))
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class OjaRuleConfig(PlasticityConfig):
    """
        Configuration for `OjaRule`.

        Parameters
        ----------
        post_tau : float or jax.Array or Initializer, default 20.0
            Decay constant of the postsynaptic trace, in ms. May be a 4-tuple, one value per
            connection type.
        eta : float, default 0.1
            Learning rate.
    """
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
class OjaRule(Plasticity):
    r"""
        Oja's rule.

        Hebbian potentiation with a decay term proportional to the weight and to the square of
        the postsynaptic trace. The decay bounds the weight vector, which plain Hebbian
        potentiation does not.

        Parameters
        ----------
        config : OjaRuleConfig
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        pre_spikes : SpikeArray
            Presynaptic spikes, after any conduction delay.
        post_spikes : SpikeArray
            Postsynaptic spikes emitted on this step.
        kernel : FloatArray
            Current synaptic weights, read from the synapse.

        Output Ports
        ------------
        kernel : FloatArray
            The updated weights, written back onto the synapse as an effect.

        Notes
        -----
        With :math:`y` the postsynaptic trace,

        .. math::
            \Delta W = \eta \left( y \, s_{\mathrm{pre}} - W y^2 \right)

        applied as :math:`W \leftarrow \max(W + \Delta t \, \Delta W, 0)`.

        References
        ----------
        .. [1] E. Oja, "A Simplified Neuron Model as a Principal Component Analyzer", Journal of
               Mathematical Biology 15(3), 267-273, 1982. https://doi.org/10.1007/BF00275687

        See Also
        --------
        HebbianRule : The same potentiation without the normalizing term.
    """
    config: OjaRuleConfig

    def __init__(self, config: OjaRuleConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray):
        # Initialize variables.
        # NOTE: Since variables used for plasticity may have very different shapes and initialization patterns
        # variables should almost always be initialized using self._initialize_variable. This method automatically
        # handles the most common initialization patterns and tries to keep the variable the smallest shape possible
        # that is still compute efficient for plasticity computation (THIS APPRAOCH IS NOT ALWAYS MEMORY EFFICIENT).
        kernel_shape = kernel.shape
        pre_shape = pre_spikes.shape
        post_shape = post_spikes.shape
        _initialize_variable_fn = lambda var: self._initialize_variable(var, shape=kernel_shape, dtype=kernel.dtype)
        _post_tau = _initialize_variable_fn(self.config.init.post_tau)
        # Broadcast shapes
        # NOTE: Most learning rules may be reexpressed as sums of general dot products; the most robuts way to 
        # implement this is by reshaping pre and post spikes to the shape of the kernel. It also makes the math more clear.
        self._pre_shape = pre_shape if pre_shape == kernel_shape else (1,) * (len(kernel_shape) - len(pre_shape)) + pre_shape
        self._post_shape = post_shape if post_shape == kernel_shape else post_shape + (1,) * (len(kernel_shape) - len(post_shape))
        # Tracers.
        self.post_trace = Tracer(self._post_shape, tau=_post_tau, scale=1/_post_tau, dtype=self._dtype, dt=self._dt)
        self.eta = Constant(self.config.eta)

    def reset(self) -> None:
        """
            Resets component state.
        """
        self.post_trace.reset()

    def _compute_kernel_update(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> jax.Array:
        """
            Computes next kernel update.
        """
        # Extract and reshape inputs
        _pre_spikes = pre_spikes.spikes.reshape(self._pre_shape)
        _post_spikes = post_spikes.spikes.reshape(self._post_shape)
        _kernel = kernel.value
        # Update and get current trace value
        post_trace = self.post_trace(_post_spikes)
        # Compute rule
        dK = self.eta.value * (
            + post_trace * _pre_spikes
            - _kernel * jnp.square(post_trace)
        )
        return jnp.clip(_kernel + self._dt * dK, min=0.0)

    def __call__(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> PlasticityOutput:
        """
            Computes the weights for the next step.

            Parameters
            ----------
            pre_spikes : SpikeArray
                Presynaptic spikes, after any conduction delay.
            post_spikes : SpikeArray
                Postsynaptic spikes emitted on this step.
            kernel : FloatArray
                Current synaptic weights.

            Returns
            -------
            PlasticityOutput
                Dictionary with one entry, ``kernel``, the updated weights.
        """
        return {
            'kernel': FloatArray(self._compute_kernel_update(pre_spikes, post_spikes, kernel))
        }
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################