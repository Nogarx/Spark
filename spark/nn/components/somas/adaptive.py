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
        Configuration for `AdaptiveSoma`.

        Each mechanism is enabled by its trigger parameter and disabled while that parameter is
        None. A disabled mechanism contributes no operations to the step.

        Parameters
        ----------
        cooldown : float or jax.Array or Initializer or None, default None
            Absolute refractory period, in ms. Enables refractoriness.
        clamp_duration : float or jax.Array or Initializer or None, default None
            Time the membrane potential is held at the reset value after a spike, in ms. Enables
            the potential clamp.
        threshold_delta : float or jax.Array or Initializer or None, default None
            Threshold increment per spike, in mV. Enables threshold adaptation.
        threshold_tau : float or jax.Array or Initializer, default 20.0
            Decay constant of the threshold offset, in ms. Read only when ``threshold_delta`` is
            set.
        adaptation_delta : float or jax.Array or Initializer or None, default None
            Adaptation current increment per spike, in pA. Enables the adaptation current.
        adaptation_tau : float or jax.Array or Initializer, default 100.0
            Decay constant of the adaptation current, in ms. Read only when ``adaptation_delta``
            is set.
        adaptation_subthreshold : float or jax.Array or Initializer, default 0.5
            Coupling of the adaptation current to the membrane potential, in nS. Read only when
            ``adaptation_delta`` is set.
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
    r"""
        Adaptation mechanisms for soma models.

        Mixin adding an absolute refractory period, a potential clamp, an adaptive threshold and
        an adaptation current to a `Soma` subclass. It must precede the model in the base list::

            class AdaptiveLeakySoma(AdaptiveSoma, LeakySoma): ...

        Parameters
        ----------
        config : AdaptiveSomaConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        current : CurrentArray
            Current delivered to the membrane, in pA. Gated off while the refractory period is active.
        inhibition_mask : BooleanMask, optional
            Marks the inhibitory units. Passed through to the model unchanged.

        Output Ports
        ------------
        spikes : SpikeArray
            Non-zero where the potential crossed the adapted threshold and no veto applied.

        Properties
        ----------
        potential : PotentialArray
            Membrane potential, as held by the model this mixin extends. Read only.

        Notes
        -----
        The mixin supplies four terms of the soma step. The membrane integration
        :math:`\Phi` comes from the model it is mixed into:

        .. math::
            I_{\mathrm{eff}} &= g_{\mathrm{in}} I + I_{\mathrm{off}} \\
            V' &= \Phi(V, I_{\mathrm{eff}}) \\
            V'' &= g_{\mathrm{dyn}} V' + (1 - g_{\mathrm{dyn}}) V_{\mathrm{reset}} \\
            s &= (V'' > \theta + \theta_{\mathrm{off}}) \wedge m

        Each trigger parameter drives the terms beside it:

        * ``cooldown`` drives :math:`g_{\mathrm{in}}` and :math:`m`.
        * ``clamp_duration`` drives :math:`g_{\mathrm{dyn}}`.
        * ``threshold_delta`` drives :math:`\theta_{\mathrm{off}}`.
        * ``adaptation_delta`` drives :math:`I_{\mathrm{off}}`.

        Refractoriness gates the input current off and vetoes the spikes for ``cooldown`` after a
        spike. The potential clamp holds the potential at ``potential_reset`` for
        ``clamp_duration``. Both share one spike counter, saturating at the longer of the two.

        The threshold offset decays exponentially and is incremented by ``threshold_delta`` on
        every spike, with :math:`\alpha_\theta = \exp(-\Delta t / \tau_\theta)`:

        .. math::
            \theta_{\mathrm{off}} \leftarrow \alpha_\theta \theta_{\mathrm{off}}
                                           + \Delta\theta \, s

        The adaptation current is subtracted from the input current, so
        :math:`I_{\mathrm{off}} = -w`. It couples to the post-reset potential and is incremented
        on every spike, with :math:`a` the subthreshold coupling and :math:`b` the increment:

        .. math::
            w \leftarrow w + \frac{\Delta t}{\tau_w} \left( -w + a V \right) + b \, s

        Mechanism state is read at the start of the step and updated in `_after_spike` from the
        spikes and the membrane potential of that step. The hooks call `super`, so a model that
        overrides the same hooks, such as `IzhikevichSoma`, keeps working when extended.

        See Also
        --------
        AdaptiveLeakySoma : Leaky membrane with these mechanisms.
        AdaptiveExponentialSoma : Exponential membrane with these mechanisms (AdEx).
        AdaptiveIzhikevichSoma : Izhikevich membrane with these mechanisms.
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
