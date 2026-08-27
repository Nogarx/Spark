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
from spark.core.backend import Constant
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.plasticity.base import Plasticity, PlasticityConfig, PlasticityOutput
from spark.nn.components.plasticity.modulated import ModulatedPlasticity, ModulatedPlasticityConfig
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ZenkeRuleConfig(PlasticityConfig):
    """
        Configuration for `ZenkeRule`.

        Parameters
        ----------
        pre_tau : float or jax.Array or Initializer, default 20.0
            Decay constant of the presynaptic trace, in ms.
        post_tau : float or jax.Array or Initializer, default 20.0
            Decay constant of the fast postsynaptic trace, in ms.
        post_slow_tau : float or jax.Array or Initializer, default 100.0
            Decay constant of the slow postsynaptic trace, in ms.
        target_tau : float or jax.Array or Initializer, default 1200000.0
            Decay constant of the weight target, in ms. Much slower than the other traces, so the
            target moves on the time scale of consolidation rather than of activity.
        a : float, default 2.0
            Weight of the triplet potentiation term.
        b : float, default -0.02
            Weight of the doublet depression term. Negative.
        c : float, default -400.0
            Weight of the heterosynaptic term. Negative.
        d : float, default 2e-05
            Weight of the transmitter-induced term.
        p : float, default 20.0
            Depth of the double-well potential that holds the weight target.
        eta : float, default 0.1
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
            'description': 'Time constant of the fast postynaptic spike train',
        })
    post_slow_tau: float | jax.Array | Initializer = dc.field(
        default = 100.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the slow postynaptic spike train',
        })
    target_tau: float | jax.Array | Initializer = dc.field(
        default = 1200000.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the kernel target',
        })
    a: float | jax.Array = dc.field(
        default = 2.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Scale factor of the triplet LTP term.',
        })
    b: float | jax.Array = dc.field(
        default = -0.02, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Scale factor of the doublet LTD term.',
        })
    c: float | jax.Array = dc.field(
        default = -400.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Scale factor of the heterosynaptic plasticity term.',
        })
    d: float | jax.Array = dc.field(
        default = 2e-5, 
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
            'description': 'Scale factor of the double-well consolidation potential.',
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
    r"""
        Zenke triplet rule with homeostatic consolidation.

        A triplet rule extended with two terms that keep the weights bounded on their own: a
        heterosynaptic term that pulls each weight towards a slowly moving target, and a
        transmitter-induced term that potentiates on presynaptic activity alone. The combination
        is stable without an external constraint on the weights.

        Parameters
        ----------
        config : ZenkeRuleConfig, optional
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
        With :math:`x` the presynaptic trace, :math:`y` and :math:`\bar{y}` the fast and slow
        postsynaptic traces and :math:`\tilde{W}` the weight target,

        .. math::
            \Delta W = \eta \left(
                a \, x \bar{y} \, s_{\mathrm{post}}
              + b \, y \, s_{\mathrm{pre}}
              + c \, (W - \tilde{W}) \, y^3 s_{\mathrm{post}}
              + d \, s_{\mathrm{pre}} \right)

        applied as :math:`W \leftarrow \max(W + \Delta t \, \Delta W, 0)`. The target follows

        .. math::
            \tilde{W} \leftarrow \tilde{W} + \lambda_{\tilde{W}} \left(
                W - p \, \tilde{W} \left(\tfrac{1}{4} - \tilde{W}\right)
                                    \left(\tfrac{1}{2} - \tilde{W}\right) - \tilde{W} \right)

        a double-well potential with minima that separate weak from strong synapses, so a weight
        settles into one of the two rather than drifting between them.

        References
        ----------
        .. [1] F. Zenke, E. J. Agnes and W. Gerstner, "Diverse Synaptic Plasticity Mechanisms
               Orchestrated to Form and Retrieve Memories in Spiking Neural Networks", Nature
               Communications 6, 6922, 2015. https://doi.org/10.1038/ncomms7922

        See Also
        --------
        HebbianRule : Pair-based potentiation without the homeostatic terms.
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

    def _kernel_delta(self, pre_spikes: SpikeArray, post_spikes: SpikeArray, kernel: FloatArray) -> jax.Array:
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
        a = self.a.value * pre_trace * post_slow_trace * _post_spikes
        # Doublet LTD
        b = self.b.value * post_trace * _pre_spikes
        # Heterosynaptic plasticity.
        c = self.c.value * (_kernel - target_trace) * (post_trace**3) * _post_spikes
        # Transmitter induced.
        d = self.d.value * _pre_spikes
        # Compute rule
        return (a + b + c + d)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ModulatedZenkeRuleConfig(ModulatedPlasticityConfig, ZenkeRuleConfig):
    """
        Configuration for `ModulatedZenkeRule`.

        Union of `ZenkeRuleConfig` and `ModulatedPlasticityConfig`. It declares no field of its
        own, so the defaults are those of `ZenkeRuleConfig`.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ModulatedZenkeRule(ModulatedPlasticity, ZenkeRule):
    r"""
        Zenke triplet rule scaled by a third factor.

        `ZenkeRule` composed with `ModulatedPlasticity`. The signal on the ``modulation`` port
        scales the whole update, the homeostatic terms included.

        Parameters
        ----------
        config : ModulatedZenkeRuleConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        modulation : FloatArray
            Third factor scaling the whole update.
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
        .. math::
            W \leftarrow \max\left(
                W + \Delta t \, (\eta M) \, \Delta W, 0 \right)

        with :math:`\Delta W` the four terms of `ZenkeRule`.

        The heterosynaptic and transmitter-induced terms are what bound the weights, and they are
        scaled along with the rest. A modulation of zero therefore suspends the homeostasis as
        well as the learning, and a sustained negative modulation removes the bound rather than
        reversing it. The weight target keeps advancing on its own in either case, since it
        follows the weights rather than the update.

        References
        ----------
        .. [1] F. Zenke, E. J. Agnes and W. Gerstner, "Diverse Synaptic Plasticity Mechanisms
               Orchestrated to Form and Retrieve Memories in Spiking Neural Networks", Nature
               Communications 6, 6922, 2015. https://doi.org/10.1038/ncomms7922

        See Also
        --------
        ZenkeRule : The same rule without the third factor.
        ModulatedPlasticity : The mixin supplying the third factor.
    """
    config: ModulatedZenkeRuleConfig

    def __init__(self, config: ModulatedZenkeRuleConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
