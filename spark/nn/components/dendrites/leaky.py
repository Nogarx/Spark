#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp
import dataclasses as dc

from spark.core.backend import Constant
from spark.core.payloads import SparkPayload
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.initializers.base import Initializer
from spark.nn.components.dendrites.base import Dendrite, DendriteConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class LeakyDendriteConfig(DendriteConfig):
    """
        Configuration for `LeakyDendrite`.

        Parameters
        ----------
        capacitance : float or jax.Array or Initializer, default 23.674
            Membrane capacitance, in pF.
        leak_conductance : float or jax.Array or Initializer, default 3.378
            Leak conductance, in nS.

        The remaining parameters are those of `DendriteConfig`.

        Notes
        -----
        The membrane is described by a capacitance and a leak conductance rather than by a time
        constant and a resistance, as the somas of the package are, because every other term of
        a dendrite is a conductance as well: the coupling, the back-propagating action potential
        and the active channels of the subclasses. The time constant of the isolated membrane
        is ``capacitance / leak_conductance``.
    """

    capacitance: float | jax.Array | Initializer = dc.field(
        default = 23.6,
        metadata = {
            'units': 'pF',
            'validators': [TypeValidator, PositiveValidator],
            'description': 'Membrane capacitance.',
        })
    leak_conductance: float | jax.Array | Initializer = dc.field(
        default = 3.4,
        metadata = {
            'units': 'nS', # 1/GΩ
            'validators': [TypeValidator, PositiveValidator],
            'description': 'Membrane leak conductance.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class LeakyDendrite(Dendrite):
    r"""
        Passive dendritic compartment.

        A leaky membrane coupled to the soma through an axial conductance and depolarized by the
        back-propagating action potential. It has no active current of its own: it filters its
        input and passes the somatic spike back into the dendritic tree.

        Parameters
        ----------
        config : LeakyDendriteConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        in_current : CurrentArray
            Current delivered to the dendrite, in pA.
        soma_potential : PotentialArray, optional
            Potential of the coupled soma, relative to the soma's rest.
        spikes : SpikeArray, optional
            Somatic spikes, driving the back-propagating action potential.

        Output Ports
        ------------
        out_current : CurrentArray
            Axial current delivered to the soma, in pA.
        plateau : FloatArray
            How far the dendrite is into a plateau, in ``[0, 1]``.

        Properties
        ----------
        potential : PotentialArray
            Dendritic membrane potential, relative to ``potential_rest``. Read only.

        Notes
        -----
        Every term of the membrane equation is a conductance times a driving force, so the
        step is integrated exactly for conductances held over the step. With
        :math:`G = g_L + g_C + g_{BAP} + \sum_i g_i` the total conductance and
        :math:`J = I + g_C V_s + g_{BAP} E_{BAP} + \sum_i g_i E_i` the reversal weighted drive,
        both in the frame where the leak reversal is zero,

        .. math::
            V_\infty = J / G, \qquad
            V_{t+1} = V_\infty + (V_t - V_\infty) \exp(-\Delta t \, G / C)

        The step is applied as :math:`V_t + (1 - e^{-\Delta t G / C}) (V_\infty - V_t)`, with the
        gain computed at single precision, so it stays accurate at half precision and small
        steps. The channel sums come from `_channel_terms`, which is empty here and filled in by
        active models such as `CalciumDendrite`. This exponential Euler step stays stable at any
        step size, which matters because the coupling makes the dendrite much faster than its
        own leak: with the default parameters the isolated time constant is 7 ms, the coupled
        one 1 ms.

        See Also
        --------
        CalciumDendrite : This membrane with the calcium and calcium activated potassium currents.
    """
    config: LeakyDendriteConfig

    def __init__(self, config: LeakyDendriteConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, **abc_args: SparkPayload) -> None:
        super().build(**abc_args)
        # Initialize variables.
        init = self.config.init
        _capacitance = init.capacitance(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _leak_conductance = init.leak_conductance(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        # Passive membrane.
        self.capacitance = Constant(_capacitance, dtype=self._dtype)
        self.leak_conductance = Constant(_leak_conductance, dtype=self._dtype)

    def _channel_terms(self, potential: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
            Conductance and reversal weighted drive of the active channels, in nS and pA.

            Returns ``(0, 0)``: the passive membrane has no channel. Active models add theirs.
        """
        zero = jnp.zeros((), dtype=self._dtype)
        return zero, zero

    def _integrate(
            self,
            potential: jax.Array,
            current: jax.Array,
            soma_potential: jax.Array | None,
            bap_conductance: jax.Array,
        ) -> jax.Array:
        """
            Membrane integration.
        """
        conductance = self.leak_conductance.value + bap_conductance
        drive = current + bap_conductance * self.bap_reversal.value
        if soma_potential is not None:
            conductance = conductance + self.coupling.value
            drive = drive + self.coupling.value * soma_potential
        channel_conductance, channel_drive = self._channel_terms(potential)
        conductance = conductance + channel_conductance
        drive = drive + channel_drive
        # Exact step for conductances held over the step.
        equilibrium = drive / conductance
        rate = self._dt * conductance.astype(jnp.float32) / self.capacitance.value.astype(jnp.float32)
        gain = -jnp.expm1(-rate).astype(self._dtype)
        return potential + gain * (equilibrium - potential)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
