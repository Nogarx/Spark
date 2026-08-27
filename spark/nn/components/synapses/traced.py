#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs

import jax
import jax.numpy as jnp
import dataclasses as dc
import spark.core.utils as utils
from spark.core.tracers import Tracer, RDTracer, RFSTracer, contract_tracer_args
from spark.core.payloads import SpikeArray, CurrentArray, SparkPayload
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator, ZeroOneValidator
from spark.nn.components.synapses.linear import LinearSynapses, LinearSynapsesConfig
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class TracedSynapsesConfig(LinearSynapsesConfig):
    """
        Configuration for `TracedSynapses`.

        Parameters
        ----------
        units : tuple of int
            Shape of the postsynaptic pool.
        kernel : jax.Array or Initializer, default SparseUniformInitializerConfig()
            Synaptic weights, in pA.
        tau : float or jax.Array or Initializer, default 5.0
            Decay constant of the postsynaptic current, in ms.
        scale : float or jax.Array, default 1.0
            Factor applied to the weighted spikes entering the trace.
        base : float or jax.Array, default 0.0
            Value the trace decays to.
    """

    tau: float | jax.Array | Initializer = dc.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Tracer decay constant.',
    })
    scale: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Tracer spike scaling.',
    })
    base: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Tracer rest value.',
    })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class TracedSynapses(LinearSynapses):
    r"""
        Linear synapse with an exponential postsynaptic current.

        The weighted spikes are passed through a single exponential trace, so one spike
        contributes a current that decays over ``tau`` rather than over a single step.

        Parameters
        ----------
        config : TracedSynapsesConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        spikes : SpikeArray
            Presynaptic spikes.

        Output Ports
        ------------
        currents : CurrentArray
            Current delivered to each postsynaptic unit, of shape ``units``.

        Properties
        ----------
        kernel : FloatArray
            Synaptic weights, in pA, of shape ``units + input_shape``. Writable, which is how a
            plasticity rule updates them.

        Notes
        -----
        With :math:`\lambda = 1 - \exp(-\Delta t / \tau)` the trace is

        .. math::
            T \leftarrow T + \lambda (T_{\mathrm{base}} - T) + c \, W s

        and the current is the sum of :math:`T` over the presynaptic axes.

        When ``tau``, ``scale`` and ``base`` are uniform along the summed axes, the trace is
        applied to the already summed current instead of to each connection. That holds one state
        entry per postsynaptic unit rather than one per weight, and gives the same result.

        See Also
        --------
        LinearSynapses : Weighted sum without a postsynaptic current.
        RDTracedSynapses : Separate rise and decay constants.
    """
    config: TracedSynapsesConfig

    # Auxiliary type hints
    current_tracer: Tracer

    def __init__(self, config: TracedSynapsesConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, **abc_args: SparkPayload):
        # Initialize shapes
        super().build(**abc_args)
        # Initialize variables.
        _tau = self.config.init.tau(key=self.get_rng_keys(1), shape=self._kernel.value.shape, dtype=self._dtype)
        # Current tracer.
        tracer_args = {
            'tau': _tau, 
            'scale': self.config.scale, 
            'base': self.config.base
        }
        reduced_args, self._contracted_tracer = contract_tracer_args(
            self._sum_axes, self._kernel.value.shape, **tracer_args,
        )
        self.current_tracer = Tracer(
            shape=utils.contracted_shape(self._kernel.value.shape, self._sum_axes)
                  if self._contracted_tracer else self._kernel.value.shape,
            **(reduced_args if self._contracted_tracer else tracer_args),
            dt=self.config.dt,
            dtype=self.config.dtype
        )

    def reset(self,) -> None:
        """
            Resets component state.
        """
        self.current_tracer.reset()

    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        currents = self._kernel.value * spikes.value
        if self._contracted_tracer:
            trace = self.current_tracer(jnp.sum(currents, axis=self._sum_axes, keepdims=True))
            return CurrentArray(trace.reshape(self._output_shape))
        return CurrentArray(jnp.sum(self.current_tracer(currents), axis=self._sum_axes))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class RDTracedSynapsesConfig(LinearSynapsesConfig):
    """
        Configuration for `RDTracedSynapses`.

        Parameters
        ----------
        units : tuple of int
            Shape of the postsynaptic pool.
        kernel : jax.Array or Initializer, default SparseUniformInitializerConfig()
            Synaptic weights, in pA.
        tau_rise : float or jax.Array or Initializer, default 1.0
            Rise constant of the postsynaptic current, in ms.
        scale_rise : float or jax.Array, default 1.0
            Factor applied to the weighted spikes entering the rise trace.
        base_rise : float or jax.Array, default 0.0
            Value the rise trace decays to.
        tau_decay : float or jax.Array or Initializer, default 5.0
            Decay constant of the postsynaptic current, in ms.
        scale_decay : float or jax.Array, default 1.0
            Factor applied to the weighted spikes entering the decay trace.
        base_decay : float or jax.Array, default 0.0
            Value the decay trace decays to.
    """

    tau_rise: float | jax.Array | Initializer = dc.field(
        default = 1.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Rise tracer decay constant.',
    })
    scale_rise: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Rise tracer spike scaling.',
    })
    base_rise: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Rise tracer rest value.',
    })
    tau_decay: float | jax.Array | Initializer = dc.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Decay tracer decay constant.',
    })
    scale_decay: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Decay tracer spike scaling.',
    })
    base_decay: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Decay tracer rest value.',
    })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class RDTracedSynapses(LinearSynapses):
    r"""
        Linear synapse with a rise-and-decay postsynaptic current.

        The weighted spikes are passed through the difference of two exponentials, which gives a
        current that rises over ``tau_rise`` and falls over ``tau_decay`` instead of jumping on
        the step a spike arrives.

        Parameters
        ----------
        config : RDTracedSynapsesConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        spikes : SpikeArray
            Presynaptic spikes.

        Output Ports
        ------------
        currents : CurrentArray
            Current delivered to each postsynaptic unit, of shape ``units``.

        Properties
        ----------
        kernel : FloatArray
            Synaptic weights, in pA, of shape ``units + input_shape``. Writable, which is how a
            plasticity rule updates them.

        Notes
        -----
        The trace is the decay exponential minus the rise exponential. The rise constant is
        coupled to the decay constant as

        .. math::
            \tau_r' = \frac{\tau_r \tau_d}{\tau_r + \tau_d}

        which is what keeps the peak at the intended height as the two constants approach each
        other.

        The contraction described in `TracedSynapses` applies here as well.

        See Also
        --------
        TracedSynapses : Single exponential.
        RFSTracedSynapses : Rise with a fast and a slow decay.
    """
    config: RDTracedSynapsesConfig

    # Auxiliary type hints
    current_tracer: RDTracer

    def __init__(self, config: RDTracedSynapsesConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, **abc_args: SparkPayload):
        # Initialize shapes
        super().build(**abc_args)
        # Initialize variables.
        _tau_rise = self.config.init.tau_rise(key=self.get_rng_keys(1), shape=self._kernel.value.shape, dtype=self._dtype)
        _tau_decay = self.config.init.tau_decay(key=self.get_rng_keys(1), shape=self._kernel.value.shape, dtype=self._dtype)
        # Current tracer.
        tracer_args = {
            'tau_rise': _tau_rise, 
            'tau_decay': _tau_decay,
            'scale_rise': self.config.scale_rise, 
            'scale_decay': self.config.scale_decay,
            'base_rise': self.config.base_rise, 
            'base_decay': self.config.base_decay,
        }
        reduced_args, self._contracted_tracer = contract_tracer_args(
            self._sum_axes, self._kernel.value.shape, **tracer_args,
        )
        self.current_tracer = RDTracer(
            shape=utils.contracted_shape(self._kernel.value.shape, self._sum_axes)
                  if self._contracted_tracer else self._kernel.value.shape,
            **(reduced_args if self._contracted_tracer else tracer_args),
            dt=self.config.dt,
            dtype=self.config.dtype
        )

    def reset(self,) -> None:
        """
            Resets component state.
        """
        self.current_tracer.reset()

    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        currents = self._kernel.value * spikes.value
        if self._contracted_tracer:
            trace = self.current_tracer(jnp.sum(currents, axis=self._sum_axes, keepdims=True))
            return CurrentArray(trace.reshape(self._output_shape))
        return CurrentArray(jnp.sum(self.current_tracer(currents), axis=self._sum_axes))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class RFSTracedSynapsesConfig(LinearSynapsesConfig):
    """
        Configuration for `RFSTracedSynapses`.

        Parameters
        ----------
        units : tuple of int
            Shape of the postsynaptic pool.
        kernel : jax.Array or Initializer, default SparseUniformInitializerConfig()
            Synaptic weights, in pA.
        alpha : float or jax.Array, default 0.8
            Weight of the fast component. The slow component takes ``1 - alpha``. Must lie in
            ``[0, 1]``.
        tau_rise : float or jax.Array or Initializer, default 1.0
            Rise constant shared by both components, in ms.
        scale_rise : float or jax.Array, default 1.0
            Factor applied to the weighted spikes entering the rise traces.
        base_rise : float or jax.Array, default 0.0
            Value the rise traces decay to.
        tau_fast_decay : float or jax.Array or Initializer, default 5.0
            Decay constant of the fast component, in ms.
        scale_fast_decay : float or jax.Array, default 1.0
            Factor applied to the weighted spikes entering the fast decay trace.
        base_fast_decay : float or jax.Array, default 0.0
            Value the fast decay trace decays to.
        tau_slow_decay : float or jax.Array or Initializer, default 50.0
            Decay constant of the slow component, in ms.
        scale_slow_decay : float or jax.Array, default 1.0
            Factor applied to the weighted spikes entering the slow decay trace.
        base_slow_decay : float or jax.Array, default 0.0
            Value the slow decay trace decays to.
    """
    alpha: float | jax.Array = dc.field(
        default = 0.8, 
        metadata = {
            'validators': [
                TypeValidator,
                ZeroOneValidator,
            ],
            'description': 'Fast-Slow blending factor.',
    })
    tau_rise: float | jax.Array | Initializer = dc.field(
        default = 1.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Rise tracer decay constant.',
    })
    scale_rise: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Rise tracer spike scaling.',
    })
    base_rise: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Rise tracer rest value.',
    })
    tau_fast_decay: float | jax.Array | Initializer = dc.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Fast-decay tracer decay constant.',
    })
    scale_fast_decay: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Fast-decay tracer spike scaling.',
    })
    base_fast_decay: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Fast-decay tracer rest value.',
    })
    tau_slow_decay: float | jax.Array | Initializer = dc.field(
        default = 50.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Slow-decay tracer decay constant.',
    })
    scale_slow_decay: float | jax.Array = dc.field(
        default = 1.0, 
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Slow-decay tracer spike scaling.',
    })
    base_slow_decay: float | jax.Array = dc.field(
        default = 0.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Slow-decay tracer rest value.',
    })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class RFSTracedSynapses(LinearSynapses):
    r"""
        Linear synapse with a two-component postsynaptic current.

        A blend of two rise-and-decay traces that share a rise constant and differ in their decay
        constants. One spike then leaves both a fast transient and a slow tail, which a single
        decay constant cannot produce.

        Parameters
        ----------
        config : RFSTracedSynapsesConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        spikes : SpikeArray
            Presynaptic spikes.

        Output Ports
        ------------
        currents : CurrentArray
            Current delivered to each postsynaptic unit, of shape ``units``.

        Properties
        ----------
        kernel : FloatArray
            Synaptic weights, in pA, of shape ``units + input_shape``. Writable, which is how a
            plasticity rule updates them.

        Notes
        -----
        .. math::
            T = \alpha T_{\mathrm{fast}} + (1 - \alpha) T_{\mathrm{slow}}

        where each component is a `RDTracedSynapses` trace built on the shared ``tau_rise``. The
        contraction described in `TracedSynapses` applies here as well.

        See Also
        --------
        RDTracedSynapses : One rise and one decay constant.
    """
    config: RFSTracedSynapsesConfig

    # Auxiliary type hints
    current_tracer: RDTracer

    def __init__(self, config: RFSTracedSynapsesConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, **abc_args: SparkPayload):
        # Initialize shapes
        super().build(**abc_args)
        # Initialize variables.
        _tau_rise = self.config.init.tau_rise(key=self.get_rng_keys(1), shape=self._kernel.value.shape, dtype=self._dtype)
        _tau_fast_decay = self.config.init.tau_fast_decay(key=self.get_rng_keys(1), shape=self._kernel.value.shape, dtype=self._dtype)
        _tau_slow_decay = self.config.init.tau_slow_decay(key=self.get_rng_keys(1), shape=self._kernel.value.shape, dtype=self._dtype)
        # Current tracer.
        tracer_args = {
            'alpha': self.config.alpha,
            'tau_rise': _tau_rise, 
            'tau_fast_decay': _tau_fast_decay, 
            'tau_slow_decay': _tau_slow_decay,
            'scale_rise': self.config.scale_rise,
            'scale_fast_decay': self.config.scale_fast_decay,
            'scale_slow_decay': self.config.scale_slow_decay,
            'base_rise': self.config.base_rise,
            'base_fast_decay': self.config.base_fast_decay,
            'base_slow_decay': self.config.base_slow_decay,
        }
        reduced_args, self._contracted_tracer = contract_tracer_args(
            self._sum_axes, self._kernel.value.shape, **tracer_args,
        )
        self.current_tracer = RFSTracer(
            shape=utils.contracted_shape(self._kernel.value.shape, self._sum_axes)
                  if self._contracted_tracer else self._kernel.value.shape,
            **(reduced_args if self._contracted_tracer else tracer_args),
            dt=self.config.dt,
            dtype=self.config.dtype
        )

    def reset(self,) -> None:
        """
            Resets component state.
        """
        self.current_tracer.reset()

    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        currents = self._kernel.value * spikes.value
        if self._contracted_tracer:
            trace = self.current_tracer(jnp.sum(currents, axis=self._sum_axes, keepdims=True))
            return CurrentArray(trace.reshape(self._output_shape))
        return CurrentArray(jnp.sum(self.current_tracer(currents), axis=self._sum_axes))
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################