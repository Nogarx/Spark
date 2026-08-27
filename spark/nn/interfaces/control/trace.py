#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs

import abc
import jax
import jax.numpy as jnp
import typing as tp
import dataclasses as dc
import spark.core.utils as utils
from spark.core.tracers import Tracer
from spark.core.payloads import FloatArray, SparkPayload
from spark.core.backend import data
from spark.core.registry import register_interface, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.control.base import ControlInterface, ControlInterfaceConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class SignalTraceOutput(tp.TypedDict):
    """
        Output ports of a signal trace.

        Attributes
        ----------
        output : FloatArray
            The trace after this step, plus whatever arrived on the ``trace`` port.
    """
    output: FloatArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SignalTraceConfig(ControlInterfaceConfig):
    """
        Base configuration for signal traces.

        Parameters
        ----------
        tau : float or jax.Array, default 10.0
            Decay constant of the trace, in ms.
        base : float or jax.Array, default 0.0
            Value the trace decays towards.
    """

    tau: float | jax.Array = dc.field(
        default = 10.0,
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Decay time constant of the trace.',
        })
    base: float | jax.Array = dc.field(
        default = 0.0,
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Value the trace decays towards.',
        })
ConfigT = tp.TypeVar("ConfigT", bound=SignalTraceConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SignalTrace(ControlInterface, tp.Generic[ConfigT]):
    r"""
        Base class for exponentially decaying traces of a signal.

        A signal that arrives at one rate and is read at another is held here rather than by the
        caller: the value is delivered as it happens and the trace is what the rest of the graph
        reads. This keeps the decay constant inside the model, where ``dt`` is known.

        Subclasses differ in the gain applied to the incoming signal, and therefore in which
        quantity is preserved when ``dt`` changes:

        * `SignalAccumulator` preserves the area of a signal delivered once.
        * `SignalAverage` preserves the level of a signal delivered on every step.

        Neither preserves both. Picking the other one rescales the trace rather than raising.

        Parameters
        ----------
        config : SignalTraceConfig
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        signal : FloatArray
            Value delivered on this step.
        trace : FloatArray, optional
            Trace arriving from an upstream `SignalTrace`, added to this one.

        Output Ports
        ------------
        output : FloatArray
            The trace after this step, plus whatever arrived on ``trace``.

        Notes
        -----
        Traces chain. Feeding the output of one into the ``trace`` port of the next adds them, so
        a port carrying signals of both kinds takes one trace of each.

        See Also
        --------
        SignalAccumulator : Area-preserving trace.
        SignalAverage : Level-preserving trace.
    """
    config: ConfigT

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    @abc.abstractmethod
    def _entry_gain(self) -> float | jax.Array:
        """
            Gain applied to the incoming signal as it enters the trace.
        """
        pass

    @property
    def _decay_rate(self) -> jax.Array:
        return -jnp.expm1(-self._dt / jnp.asarray(self.config.tau, dtype=jnp.float32))

    def build(self, signal: FloatArray, trace: FloatArray | None = None) -> None:
        # Initialize shapes
        self.shape = utils.validate_shape(signal.shape)
        # Initialize variables.
        self.trace = data(Tracer(
            self.shape,
            tau=self.config.tau,
            base=self.config.base,
            scale=self._entry_gain(),
            dt=self._dt, dtype=self._dtype,
        ))

    def reset(self) -> None:
        """
            Resets component state.
        """
        self.trace.reset()

    def __call__(self, signal: FloatArray, trace: FloatArray | None = None) -> SignalTraceOutput:
        """
            Advances the trace with the incoming signal.

            Parameters
            ----------
            signal : FloatArray
                Value delivered on this step.
            trace : FloatArray, optional
                Trace arriving from an upstream `SignalTrace`, added to this one.

            Returns
            -------
            SignalTraceOutput
                Dictionary with one entry, ``output``, the trace after this step.
        """
        own = self.trace(signal.value)
        return {
            'output': FloatArray(own if trace is None else own + trace.value)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class SignalAccumulatorConfig(SignalTraceConfig):
    """
        Configuration for `SignalAccumulator`.

        Parameters
        ----------
        tau : float or jax.Array, default 10.0
            Decay constant of the trace, in ms.
        base : float or jax.Array, default 0.0
            Value the trace decays towards.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class SignalAccumulator(SignalTrace):
    r"""
        Area-preserving trace of a signal.

        The signal enters the trace unscaled, so a one-off delivery leaves a jump of its full
        magnitude that then decays. Use this for events: a reward, a cue, anything delivered on
        the step it happens and on no other.

        Parameters
        ----------
        config : SignalAccumulatorConfig
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        signal : FloatArray
            Value delivered on this step.
        trace : FloatArray, optional
            Trace arriving from an upstream `SignalTrace`, added to this one.

        Output Ports
        ------------
        output : FloatArray
            The trace after this step, plus whatever arrived on ``trace``.

        Notes
        -----
        With :math:`\lambda = 1 - \exp(-\Delta t / \tau)`,

        .. math::
            T \leftarrow T + \lambda (T_{\mathrm{base}} - T) + x

        A signal delivered on every step accumulates to :math:`x / \lambda`, which grows as
        ``dt`` shrinks. Use `SignalAverage` for that case.

        See Also
        --------
        SignalAverage : Level-preserving trace.
    """
    config: SignalAccumulatorConfig

    def __init__(self, config: SignalAccumulatorConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def _entry_gain(self) -> float | jax.Array:
        return 1.0

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class SignalAverageConfig(SignalTraceConfig):
    """
        Configuration for `SignalAverage`.

        Parameters
        ----------
        tau : float or jax.Array, default 10.0
            Decay constant of the trace, in ms.
        base : float or jax.Array, default 0.0
            Value the trace decays towards.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class SignalAverage(SignalTrace):
    r"""
        Level-preserving trace of a signal.

        The signal is scaled by the decay rate as it enters the trace, so a constant input settles
        at that constant rather than at a multiple of it. Use this for signals present on every
        step: a firing rate, a sensor reading, a running error.

        Parameters
        ----------
        config : SignalAverageConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        signal : FloatArray
            Value delivered on this step.
        trace : FloatArray, optional
            Trace arriving from an upstream `SignalTrace`, added to this one.

        Output Ports
        ------------
        output : FloatArray
            The trace after this step, plus whatever arrived on ``trace``.

        Notes
        -----
        With :math:`\lambda = 1 - \exp(-\Delta t / \tau)`,

        .. math::
            T \leftarrow T + \lambda (T_{\mathrm{base}} - T) + \lambda x

        which is the exponential moving average of :math:`x`. A one-off delivery leaves a jump of
        :math:`\lambda x`, which shrinks with ``dt``. Use `SignalAccumulator` for that case.

        See Also
        --------
        SignalAccumulator : Area-preserving trace.
    """
    config: SignalAverageConfig

    def __init__(self, config: SignalAverageConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def _entry_gain(self) -> float | jax.Array:
        return self._decay_rate

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
