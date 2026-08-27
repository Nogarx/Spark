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
from spark.core.payloads import SparkPayload
from spark.core.backend import Variable, Constant
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator
from spark.nn.components.somas.base import Soma, SomaConfig
from spark.nn.initializers.base import Initializer
from spark.nn.components.somas.adaptive import AdaptiveSoma, AdaptiveSomaConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class IzhikevichSomaConfig(SomaConfig):
    """
        Configuration for `IzhikevichSoma`.

        Parameters
        ----------
        potential_rest : float or jax.Array or Initializer, default -65.0
            Membrane rest potential, in mV. The membrane and the recovery variable start here.
        potential_reset : float or jax.Array or Initializer, default -65.0
            Membrane potential after a spike, in mV. Parameter ``c`` of the Izhikevich model.
        resistance : float or jax.Array or Initializer, default 0.1
            Membrane resistance, in GΩ.
        threshold : float or jax.Array or Initializer, default 30.0
            Peak potential at which a spike is registered, in mV.
        recovery_timescale : float or jax.Array or Initializer, default 0.02
            Time scale of the recovery variable. Parameter ``a`` of the Izhikevich model.
        recovery_sensitivity : float or jax.Array or Initializer, default 0.2
            Sensitivity of the recovery variable to subthreshold fluctuations of the membrane
            potential. Parameter ``b`` of the Izhikevich model.
        recovery_update : float or jax.Array or Initializer, default 2
            Recovery increment per spike. Parameter ``d`` of the Izhikevich model.
    """

    potential_rest: float | jax.Array  | Initializer = dc.field(
        default = -65.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane rest potential.',
        })
    potential_reset: float | jax.Array | Initializer = dc.field(
        default = -65.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane after spike reset potential. ' +
                           'C parameter for the Izhikevich model.',
        })
    resistance: float | jax.Array | Initializer= dc.field(
        default = 0.1,
        metadata = {
            'units': 'GΩ', # [1/nS]
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane resistance.',
        })
    threshold: float | jax.Array | Initializer = dc.field(
        default = 30.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Action potential peak value.',
        })
    recovery_timescale: float | jax.Array | Initializer = dc.field(
        default = 0.02,
        metadata = {
            'units': '',
            'validators': [
                TypeValidator,
            ],
            'description': 'Time scale of the recovery variable. ' +
                           'A parameter for the Izhikevich model.',
        })
    recovery_sensitivity: float | jax.Array | Initializer = dc.field(
        default = 0.2,
        metadata = {
            'units': '',
            'validators': [
                TypeValidator,
            ],
            'description': 'Sensitivity of the recovery variable to the subthreshold fluctuations of the membrane potential. ' +
                           'B parameter for the Izhikevich model.',
        })
    recovery_update: float | jax.Array | Initializer = dc.field(
        default = 2,
        metadata = {
            'units': '',
            'validators': [
                TypeValidator,
            ],
            'description': 'Recovery increment after spike. ' +
                           'D parameter for the Izhikevich model.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class IzhikevichSoma(Soma):
    r"""
        Izhikevich soma.

        A quadratic membrane paired with a recovery variable. The pair reproduces a wide range of
        firing patterns, selected through ``recovery_timescale``, ``recovery_sensitivity``,
        ``recovery_update`` and ``potential_reset``. A spike is registered when the potential
        reaches ``threshold``, after which the potential is set to ``potential_reset`` and the
        recovery variable is incremented.

        Parameters
        ----------
        config : IzhikevichSomaConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        current : CurrentArray
            Current delivered to the membrane, in pA.
        inhibition_mask : BooleanMask, optional
            Marks the inhibitory units. Supplied by the enclosing `Neuron`.

        Output Ports
        ------------
        spikes : SpikeArray
            Non-zero where the potential crossed the threshold on this step.

        Properties
        ----------
        potential : PotentialArray
            Membrane potential, in mV. Read only.

        Notes
        -----
        Unlike the other somas in this package, potentials are stored in absolute mV rather than
        relative to rest: the quadratic term is not invariant under a shift of the potential.

        With :math:`u` the recovery variable, the step is

        .. math::
            V_{t+1} &= V_t + \Delta t \left( 0.04 V_t^2 + 5 V_t + 140 - u_t + R I_t \right) \\
            u_{t+1} &= u_t + \Delta t \, a \left( b V_{t+1} - u_t \right)

        and a spike adds :math:`d` to :math:`u`. The constants 0.04, 5 and 140 are those of the
        original model and assume :math:`V` in mV and :math:`\Delta t` in ms.

        The recovery variable reads the potential produced by the same step, so it is updated in
        `_post_integrate` and `_after_spike` rather than alongside the membrane. Those hooks call
        `super`, which leaves `AdaptiveSoma` free to extend the model.

        References
        ----------
        .. [1] E. M. Izhikevich, "Simple Model of Spiking Neurons", IEEE Transactions on Neural
               Networks 14(6), 1569-1572, 2003. https://doi.org/10.1109/TNN.2003.820440

        See Also
        --------
        AdaptiveIzhikevichSoma : This model with the adaptation mechanisms.
    """
    config: IzhikevichSomaConfig

    def __init__(self, config: IzhikevichSomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, **abc_args: SparkPayload) -> None:
        super().build(**abc_args)
        # Initialize variables.
        _potential_rest = self.config.init.potential_rest(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _potential_reset = self.config.init.potential_reset(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _recovery_update = self.config.init.recovery_update(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _recovery_timescale = self.config.init.recovery_timescale(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _recovery_sensitivity = self.config.init.recovery_sensitivity(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _resistance = self.config.init.resistance(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _threshold = self.config.init.threshold(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        # Membrane.
        self.potential_rest = Constant(_potential_rest, dtype=self._dtype)
        self.potential_reset = Constant(_potential_reset, dtype=self._dtype)
        # Recovery.
        self.recovery_update = Constant(_recovery_update, dtype=self._dtype)
        self.recovery_timescale = Constant(_recovery_timescale, dtype=self._dtype)
        self.recovery_sensitivity = Constant(_recovery_sensitivity, dtype=self._dtype)
        self.recovery = Variable(self._rest_recovery(), dtype=self._dtype)
        # Conductance.
        self.resistance = Constant(_resistance, dtype=self._dtype) # Current is in pA for stability
        # Threshold.
        self.threshold = Constant(_threshold, dtype=self._dtype)
        # Membrane potential starts at rest rather than at zero.
        self._potential.value = self._rest_potential()

    def _rest_potential(self) -> jax.Array:
        return jnp.broadcast_to(self.potential_rest.value, self.units).astype(self._dtype)

    def _rest_recovery(self) -> jax.Array:
        return jnp.broadcast_to(
            self.recovery_sensitivity.value * self.potential_rest.value, self.units,
        ).astype(self._dtype)

    def reset(self) -> None:
        """
            Resets component state.
        """
        self._potential.value = self._rest_potential()
        self.recovery.value = self._rest_recovery()

    def _integrate(self, potential: jax.Array, current: jax.Array) -> jax.Array:
        """
            Membrane integration.
        """
        return potential + self._dt * (
            + 0.04 * potential * potential
            + 5.0 * potential
            + 140.0
            - self.recovery.value
            + self.resistance.value * current
        )

    def _post_integrate(self, potential: jax.Array) -> jax.Array:
        """
            The recovery variable follows the potential within the same step, which is why it
            is updated here rather than after the spikes are out.
        """
        self.recovery.value = self.recovery.value + self._dt * self.recovery_timescale.value * (
            self.recovery_sensitivity.value * potential - self.recovery.value
        )
        return potential

    def _after_spike(self, spikes: jax.Array) -> None:
        """
            Spike triggered increment of the recovery variable.
        """
        self.recovery.value = self.recovery.value + spikes * self.recovery_update.value

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class AdaptiveIzhikevichSomaConfig(AdaptiveSomaConfig, IzhikevichSomaConfig):
    """
        Configuration for `AdaptiveIzhikevichSoma`.

        Union of `IzhikevichSomaConfig` and `AdaptiveSomaConfig`. It declares no field of its own.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class AdaptiveIzhikevichSoma(AdaptiveSoma, IzhikevichSoma):
    """
        Izhikevich soma with the adaptation mechanisms.

        `IzhikevichSoma` composed with `AdaptiveSoma`, which adds an absolute refractory period, a
        potential clamp, an adaptive threshold and an adaptation current. The recovery variable of
        the Izhikevich model is unaffected and keeps its own dynamics.

        Parameters
        ----------
        config : AdaptiveIzhikevichSomaConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        current : CurrentArray
            Current delivered to the membrane, in pA.
        inhibition_mask : BooleanMask, optional
            Marks the inhibitory units. Supplied by the enclosing `Neuron`.

        Output Ports
        ------------
        spikes : SpikeArray
            Non-zero where the potential crossed the threshold on this step.

        Properties
        ----------
        potential : PotentialArray
            Membrane potential, in mV. Read only.

        See Also
        --------
        IzhikevichSoma : The membrane integration, without the mechanisms.
        AdaptiveSoma : The mechanisms and their equations.
    """
    config: AdaptiveIzhikevichSomaConfig

    def __init__(self, config: AdaptiveIzhikevichSomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################