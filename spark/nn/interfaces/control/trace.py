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
       Generic signal trace model output spec.
    """
    output: FloatArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SignalTraceConfig(ControlInterfaceConfig):
    """
        Abstract signal trace configuration class.
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
    """
        Abstract exponentially decaying trace of a signal.

        A model driven by an environment sees values that arrive at one rate and matter at
        another: a reward, a context cue, a neuromodulator release. Holding such a value and
        decaying it between calls puts a piece of the model in the training loop, and makes the
        decay constant a number the caller has to keep consistent with the model's dt. A trace
        owns it instead, so the signal is delivered as it happens and the trace is what the rest
        of the graph reads.

        What separates the two implementations is which quantity survives a change of dt, and
        that is why they are separate components rather than one with a switch:

            SignalAccumulator   conserves the area of a signal delivered once
            SignalAverage       conserves the level of a signal delivered on every step

        Neither conserves both, and choosing the wrong one does not fail loudly: it rescales the
        signal the moment dt changes.

        Traces chain. Feeding the output of one into the "trace" port of the next adds them, so
        a port carrying signals of both kinds takes one of each, rather than a single trace with
        a correction factor applied to whichever signal lost.
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
            Advances the trace with the incoming signal and adds whatever arrived from upstream.
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
        SignalAccumulator configuration class.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class SignalAccumulator(SignalTrace):
    """
        Leaky accumulation of a signal.

        The signal enters at full amplitude and decays from there, so a value delivered on one
        step and followed by silence reproduces exactly the decaying schedule a caller would
        otherwise have written out by hand. What is conserved is the area under a single
        delivery, which approaches tau times its amplitude as dt shrinks, so an event keeps its
        meaning when the integration step changes.

        A signal delivered on every step accumulates rather than settling on itself, and the
        level it reaches grows as dt shrinks. Use SignalAverage for that.

        Init:
            tau: float [ms]
            base: float

        Input:
            signal: FloatArray
            trace: FloatArray | None

        Output:
            output: FloatArray
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
        SignalAverage configuration class.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class SignalAverage(SignalTrace):
    """
        Exponential moving average of a signal.

        A signal delivered on every step settles on itself whatever the integration step, so a
        level keeps its meaning when dt changes. This is also the safer of the two at reduced
        precision: a level is often the smaller of the signals sharing a port, and this trace is
        the one that does not ask it to carry a correction factor.

        A signal delivered once enters attenuated by one minus the decay per step, and the area
        it leaves behind shrinks with dt. Use SignalAccumulator for that.

        Init:
            tau: float [ms]
            base: float

        Input:
            signal: FloatArray
            trace: FloatArray | None

        Output:
            output: FloatArray
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
