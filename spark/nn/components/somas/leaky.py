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
from spark.core.backend import Constant
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.somas.base import Soma, SomaConfig
from spark.nn.initializers.base import Initializer
from spark.nn.components.somas.adaptive import AdaptiveSoma, AdaptiveSomaConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class LeakySomaConfig(SomaConfig):
    """
        Configuration for `LeakySoma`.

        Parameters
        ----------
        potential_rest : float or jax.Array or Initializer, default -60.0
            Membrane rest potential, in mV.
        potential_reset : float or jax.Array or Initializer, default -50.0
            Membrane potential after a spike, in mV.
        potential_tau : float or jax.Array or Initializer, default 20.0
            Membrane potential decay constant, in ms.
        resistance : float or jax.Array or Initializer, default 0.1
            Membrane resistance, in GΩ.
        threshold : float or jax.Array or Initializer, default -40.0
            Spike threshold, in mV.
    """

    potential_rest: float | jax.Array | Initializer = dc.field(
        default = -60.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane rest potential.',
        })
    potential_reset: float | jax.Array | Initializer = dc.field(
        default = -50.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane after spike reset potential.',
        })
    potential_tau: float | jax.Array | Initializer = dc.field(
        default = 20.0,
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Membrane potential decay constant.',
        })
    resistance: float | jax.Array | Initializer = dc.field(
        default = 0.1,
        metadata = {
            'units': 'GΩ', # [1/nS]
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane resistance.',
        })
    threshold: float | jax.Array | Initializer = dc.field(
        default = -40.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Action potential threshold base value.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class LeakySoma(Soma):
    r"""
        Leaky integrate-and-fire soma.

        Parameters
        ----------
        config : LeakySomaConfig
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
            Membrane potential, relative to ``potential_rest``. Read only.

        Notes
        -----
        Potentials are stored relative to ``potential_rest``, so a stored value of zero is rest.
        The membrane is integrated in closed form, with
        :math:`\alpha = \exp(-\Delta t / \tau_V)`:

        .. math::
            V_{t+1} = \alpha V_t + (1 - \alpha) R I_t

        References
        ----------
        .. [1] W. Gerstner, W. M. Kistler, R. Naud and L. Paninski, "Neuronal Dynamics: From
               Single Neurons to Networks and Models of Cognition", Chapter 1.3, Integrate-And-Fire
               Models. https://neuronaldynamics.epfl.ch/online/Ch1.S3.html

        See Also
        --------
        AdaptiveLeakySoma : This model with the adaptation mechanisms.
    """
    config: LeakySomaConfig

    def __init__(self, config: LeakySomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    # NOTE: potential_rest is substracted to potential related terms to rebase potential at zero.
    def build(self, **abc_args: SparkPayload) -> None:
        super().build(**abc_args)
        # Initialize variables.
        _potential_rest = self.config.init.potential_rest(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _potential_reset = self.config.init.potential_reset(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _potential_tau = self.config.init.potential_tau(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _resistance = self.config.init.resistance(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _threshold = self.config.init.threshold(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        # Membrane. Substract potential_rest to potential related terms to rebase potential at zero.
        self.potential_rest = Constant(_potential_rest, dtype=self._dtype)
        self.potential_reset = Constant(_potential_reset - _potential_rest, dtype=self._dtype)
        self.potential_decay = Constant(jnp.exp(-self._dt / _potential_tau), dtype=self._dtype)
        self.potential_gain = Constant((1 - self.potential_decay.value), dtype=self._dtype)
        # Conductance.
        self.resistance = Constant(_resistance, dtype=self._dtype) # Current is in pA for stability
        # Threshold.
        self.threshold = Constant(_threshold - _potential_rest, dtype=self._dtype)

    def _integrate(self, potential: jax.Array, current: jax.Array) -> jax.Array:
        """
            Membrane integration.
        """
        return (
            + self.potential_decay.value * potential
            + self.potential_gain.value * self.resistance.value * current
        )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class AdaptiveLeakySomaConfig(AdaptiveSomaConfig, LeakySomaConfig):
    """
        Configuration for `AdaptiveLeakySoma`.

        Union of `LeakySomaConfig` and `AdaptiveSomaConfig`. It declares no field of its own.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class AdaptiveLeakySoma(AdaptiveSoma, LeakySoma):
    """
        Leaky integrate-and-fire soma with the adaptation mechanisms.

        `LeakySoma` composed with `AdaptiveSoma`, which adds an absolute refractory period, a
        potential clamp, an adaptive threshold and an adaptation current. Each is enabled by its
        trigger parameter and costs nothing while that parameter is None.

        Parameters
        ----------
        config : AdaptiveLeakySomaConfig, optional
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
            Membrane potential, relative to ``potential_rest``. Read only.

        See Also
        --------
        LeakySoma : The membrane integration, without the mechanisms.
        AdaptiveSoma : The mechanisms and their equations.
    """
    config: AdaptiveLeakySomaConfig

    def __init__(self, config: AdaptiveLeakySomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################