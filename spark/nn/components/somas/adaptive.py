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
from spark.core.tracers import Tracer
from spark.core.backend import Variable, Constant, data
from spark.core.payloads import SparkPayload
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.base import ComponentConfig
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class AdaptiveSomaConfig(ComponentConfig):
    """
        Configuration for AdaptiveSoma.

        A mechanism is enabled when its trigger parameter is not None. The triggers are
        "cooldown", "clamp_duration", "threshold_delta" and "adaptation_delta".
    """

    cooldown: float | jax.Array | Initializer | None = dc.field(
        default = None,
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Soma refractory period. Enables the absolute refractory period.',
        })
    clamp_duration: float | jax.Array | Initializer | None = dc.field(
        default = None,
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
            ],
            'description': 'Time the membrane potential is held at the reset value after a spike. '
                           'Enables the potential clamp.',
        })
    threshold_delta: float | jax.Array | Initializer | None = dc.field(
        default = None,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Adaptive action potential threshold after spike increment. Enables threshold adaptation.',
        })
    threshold_tau: float | jax.Array | Initializer = dc.field(
        default = 20.0,
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Adaptive action potential threshold decay constant.',
        })
    adaptation_delta: float | jax.Array | Initializer | None = dc.field(
        default = None,
        metadata = {
            'units': 'pA',
            'validators': [
                TypeValidator,
            ],
            'description': 'Adaptation current after spike increment. Enables the adaptation current.',
        })
    adaptation_tau: float | jax.Array | Initializer = dc.field(
        default = 100.0,
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Adaptation current decay constant.',
        })
    adaptation_subthreshold: float | jax.Array | Initializer = dc.field(
        default = 0.5,
        metadata = {
            'units': 'nS', # 1/GΩ
            'validators': [
                TypeValidator,
            ],
            'description': 'Scale factor of the subthreshold adaptation (potential-based adaptation).',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class AdaptiveSoma:
    """
        Adaptation mechanisms for soma models.

        Mixin adding an absolute refractory period, a potential clamp, an adaptive threshold
        and an adaptation current to a Soma subclass. It must precede the model in the base
        list:

            class AdaptiveLeakySoma(AdaptiveSoma, LeakySoma): ...

        The mixin supplies four terms of the soma step. The membrane integration Phi is
        provided by the model:

            I_eff = input_gain * I + current_offset
            V'    = Phi(V, I_eff)
            V     = dynamics_gain * V' + (1 - dynamics_gain) * V_reset
            s     = (V > threshold + threshold_offset) and spike_mask

        Each mechanism is enabled by its trigger parameter and drives the terms beside it:

            cooldown          input_gain, spike_mask
            clamp_duration    dynamics_gain
            threshold_delta   threshold_offset
            adaptation_delta  current_offset

        A disabled mechanism contributes no operations. Refraction and the potential clamp
        share a single spike counter.

        Mechanism state is read at the start of the step and updated in _after_spike from the
        spikes and membrane potential of that step. The hooks call super(), so a model that
        overrides the same hooks, such as IzhikevichSoma, keeps working when extended.

        Init:
            cooldown: float | jax.Array | None
            clamp_duration: float | jax.Array | None
            threshold_delta: float | jax.Array | None
            threshold_tau: float | jax.Array
            adaptation_delta: float | jax.Array | None
            adaptation_tau: float | jax.Array
            adaptation_subthreshold: float | jax.Array
    """
    config: AdaptiveSomaConfig

    def __init__(self, config: AdaptiveSomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # A mechanism is active when its trigger parameter carries a value.
        self.has_refraction = self.config.cooldown is not None
        self.has_potential_clamp = self.config.clamp_duration is not None
        self.has_adaptive_threshold = self.config.threshold_delta is not None
        self.has_adaptation_current = self.config.adaptation_delta is not None

    def build(self, **abc_args: SparkPayload) -> None:
        super().build(**abc_args)
        init = self.config.init
        _steps = lambda duration: jnp.round(duration / self._dt).astype(jnp.int32)

        # Refraction and the potential clamp share one counter.
        durations = []
        if self.has_refraction:
            _cooldown = init.cooldown(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
            self.cooldown = Constant(_steps(_cooldown), dtype=jnp.int32)
            durations.append(self.cooldown.value)
        if self.has_potential_clamp:
            _clamp_duration = init.clamp_duration(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
            self.clamp_duration = Constant(_steps(_clamp_duration), dtype=jnp.int32)
            durations.append(self.clamp_duration.value)
        if durations:
            # Saturates at the longest duration in use.
            ceiling = durations[0]
            for duration in durations[1:]:
                ceiling = jnp.maximum(ceiling, duration)
            self.spike_counter_ceiling = Constant(ceiling, dtype=jnp.int32)
            self.spike_counter = Variable(self._recovered_counter(), dtype=jnp.int32)

        # Adaptive threshold. The tracer holds the threshold offset, not the threshold.
        if self.has_adaptive_threshold:
            _threshold_tau = init.threshold_tau(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
            _threshold_delta = init.threshold_delta(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
            self.threshold_trace = data(Tracer(
                self.units,
                tau=_threshold_tau,
                base=0.0,
                scale=_threshold_delta,
                dt=self._dt, dtype=self._dtype,
            ))

        # Spike triggered and subthreshold adaptation current.
        if self.has_adaptation_current:
            _adaptation_tau = init.adaptation_tau(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
            _adaptation_delta = init.adaptation_delta(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
            _adaptation_subthreshold = init.adaptation_subthreshold(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
            self.adaptation_current = Variable(jnp.zeros(self.units, dtype=self._dtype), dtype=self._dtype)
            self.adaptation_delta = Constant(_adaptation_delta, dtype=self._dtype)
            self.adaptation_subthreshold = Constant(_adaptation_subthreshold, dtype=self._dtype)
            self.adaptation_scale = Constant(self._dt / _adaptation_tau, dtype=self._dtype)

    def _recovered_counter(self) -> jax.Array:
        """
            Spike counter value at full recovery.
        """
        return jnp.broadcast_to(self.spike_counter_ceiling.value, self.units).astype(jnp.int32)

    def reset(self) -> None:
        """
            Resets component state.
        """
        super().reset()
        if self.has_refraction or self.has_potential_clamp:
            self.spike_counter.value = self._recovered_counter()
        if self.has_adaptive_threshold:
            self.threshold_trace.reset()
        if self.has_adaptation_current:
            self.adaptation_current.value = jnp.zeros(self.units, dtype=self._dtype)

    def _effective_current(self, current: jax.Array) -> jax.Array:
        """
            Applies input_gain and current_offset.

            The input is gated off while the refractory period is active. The adaptation
            current is subtracted from the result.
        """
        current = super()._effective_current(current)
        if self.has_refraction:
            ready = jnp.greater_equal(self.spike_counter.value, self.cooldown.value)
            current = ready.astype(self._dtype) * current
        if self.has_adaptation_current:
            current = current - self.adaptation_current.value
        return current

    def _post_integrate(self, potential: jax.Array) -> jax.Array:
        """
            Applies dynamics_gain.

            While the clamp is active the membrane potential is held at the reset value and
            the integrated value is discarded.
        """
        if self.has_potential_clamp:
            released = jnp.greater_equal(self.spike_counter.value, self.clamp_duration.value)
            released = released.astype(self._dtype)
            potential = released * potential + (1 - released) * self.potential_reset.value
        return super()._post_integrate(potential)

    def _effective_threshold(self) -> jax.Array:
        """
            Applies threshold_offset.
        """
        threshold = super()._effective_threshold()
        if self.has_adaptive_threshold:
            threshold = threshold + self.threshold_trace.value
        return threshold

    def _spike_mask(self) -> jax.Array | None:
        """
            Applies spike_mask. Spikes are vetoed while the refractory period is active.
        """
        mask = super()._spike_mask()
        if not self.has_refraction:
            return mask
        ready = jnp.greater_equal(self.spike_counter.value, self.cooldown.value)
        return ready if mask is None else jnp.logical_and(mask, ready)

    def _after_spike(self, spikes: jax.Array) -> None:
        """
            Updates the mechanism state from the completed step.
        """
        super()._after_spike(spikes)
        if self.has_refraction or self.has_potential_clamp:
            self.spike_counter.value = jnp.where(
                spikes.astype(jnp.bool),
                jnp.zeros((), dtype=jnp.int32),
                jnp.minimum(self.spike_counter.value + 1, self.spike_counter_ceiling.value),
            )
        if self.has_adaptive_threshold:
            self.threshold_trace(spikes.astype(self._dtype))
        if self.has_adaptation_current:
            # self._potential holds the post-reset potential.
            self.adaptation_current.value += self.adaptation_scale.value * (
                - self.adaptation_current.value
                + self.adaptation_subthreshold.value * self._potential.value
            ) + self.adaptation_delta.value * spikes.astype(self._dtype)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
