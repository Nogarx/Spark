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
class ExponentialSomaConfig(SomaConfig):
    """
        Configuration for `ExponentialSoma`.

        Parameters
        ----------
        potential_rest : float or jax.Array or Initializer, default -70.0
            Membrane rest potential, in mV.
        potential_reset : float or jax.Array or Initializer, default -51.0
            Membrane potential after a spike, in mV.
        potential_tau : float or jax.Array or Initializer, default 5.0
            Membrane potential decay constant, in ms.
        resistance : float or jax.Array or Initializer, default 0.5
            Membrane resistance, in GΩ.
        threshold : float or jax.Array or Initializer, default -30.0
            Potential at which a spike is registered, in mV. This is the cutoff of the exponential
            upswing, not the point where firing begins.
        rheobase_threshold : float or jax.Array or Initializer, default -50.0
            Rheobase threshold, in mV. The potential above which the exponential term dominates
            and the upswing becomes irreversible.
        spike_slope : float or jax.Array or Initializer, default 2.0
            Sharpness of spike initiation, in mV. Smaller values approach a hard threshold.
    """
    potential_rest: float | jax.Array | Initializer = dc.field(
        default = -70.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane rest potential.',
        })
    potential_reset: float | jax.Array | Initializer = dc.field(
        default = -51.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane after spike reset potential.',
        })
    potential_tau: float | jax.Array | Initializer = dc.field(
        default = 5.0,
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Membrane potential decay constant.',
        })
    resistance: float | jax.Array | Initializer = dc.field(
        default = 0.5,
        metadata = {
            'units': 'GΩ', # [1/nS]
            'validators': [
                TypeValidator,
            ],
            'description': 'Membrane resistance.',
        })
    threshold: float | jax.Array | Initializer = dc.field(
        default = -30.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Action potential threshold base value.',
        })
    rheobase_threshold: float | jax.Array | Initializer = dc.field(
        default = -50.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Rheobase threshold (exponential term threshold).',
        })
    spike_slope: float | jax.Array | Initializer = dc.field(
        default = 2.0,
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ],
            'description': 'Sharpness of action potential initiation.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ExponentialSoma(Soma):
    r"""
        Exponential integrate-and-fire soma.

        A leaky membrane with an added exponential term, which reproduces the upswing of a spike
        rather than firing on a hard threshold crossing. A spike is registered when the potential
        reaches ``threshold``, after which the potential is set to ``potential_reset``.

        Note that this is not the adaptive exponential (AdEx) model.

        Parameters
        ----------
        config : ExponentialSomaConfig, optional
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
        With :math:`\Delta_T` the slope factor and :math:`V_{rh}` the rheobase threshold, the step
        is a forward Euler integration:

        .. math::
            V_{t+1} = V_t + \frac{\Delta t}{\tau_V} \left(
                -V_t + \Delta_T \exp\!\left(\frac{V_t - V_{rh}}{\Delta_T}\right) + R I_t \right)

        References
        ----------
        .. [1] N. Fourcaud-Trocmé, D. Hansel, C. van Vreeswijk and N. Brunel, "How Spike Generation
               Mechanisms Determine the Neuronal Response to Fluctuating Inputs", Journal of
               Neuroscience 23(37), 11628-11640, 2003.
               https://www.jneurosci.org/content/23/37/11628
        .. [2] W. Gerstner, W. M. Kistler, R. Naud and L. Paninski, "Neuronal Dynamics: From
               Single Neurons to Networks and Models of Cognition", Chapter 5.2, Exponential
               Integrate-and-Fire Model. https://neuronaldynamics.epfl.ch/online/Ch5.S2.html

        See Also
        --------
        AdaptiveExponentialSoma : This model with the adaptation mechanisms (AdEx).
    """
    config: ExponentialSomaConfig

    def __init__(self, config: ExponentialSomaConfig | None = None, **kwargs) -> None:
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
        _rheobase_threshold = self.config.init.rheobase_threshold(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _spike_slope = self.config.init.spike_slope(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        # Membrane. Substract potential_rest to potential related terms to rebase potential at zero.
        self.potential_rest = Constant(_potential_rest, dtype=self._dtype)
        self.potential_reset = Constant(_potential_reset - _potential_rest, dtype=self._dtype)
        self.potential_scale = Constant(self._dt / _potential_tau, dtype=self._dtype)
        # Conductance.
        self.resistance = Constant(_resistance, dtype=self._dtype)
        # Threshold.
        self.threshold = Constant(_threshold - _potential_rest, dtype=self._dtype)
        self.rheobase_threshold = Constant(_rheobase_threshold - _potential_rest, dtype=self._dtype)
        # Spike slope.
        self.spike_slope = Constant(_spike_slope, dtype=self._dtype)

    def _integrate(self, potential: jax.Array, current: jax.Array) -> jax.Array:
        """
            Membrane integration.
        """
        return potential + self.potential_scale.value * (
            - potential
            + self.spike_slope.value * jnp.exp((potential - self.rheobase_threshold.value) / self.spike_slope.value)
            + self.resistance.value * current
        )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class AdaptiveExponentialSomaConfig(AdaptiveSomaConfig, ExponentialSomaConfig):
    """
        Configuration for `AdaptiveExponentialSoma`.

        Union of `ExponentialSomaConfig` and `AdaptiveSomaConfig`. It declares no field of its own.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class AdaptiveExponentialSoma(AdaptiveSoma, ExponentialSoma):
    """
        Adaptive exponential integrate-and-fire soma (AdEx).

        `ExponentialSoma` composed with `AdaptiveSoma`. Setting ``adaptation_delta`` and
        ``adaptation_subthreshold`` gives the adaptation current of the AdEx model; the refractory
        period, the potential clamp and the adaptive threshold are available on the same terms as
        for any other soma.

        Parameters
        ----------
        config : AdaptiveExponentialSomaConfig, optional
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

        References
        ----------
        .. [1] R. Brette and W. Gerstner, "Adaptive Exponential Integrate-and-Fire Model as an
               Effective Description of Neuronal Activity", Journal of Neurophysiology 94(5),
               3637-3642, 2005. https://doi.org/10.1152/jn.00686.2005

        See Also
        --------
        ExponentialSoma : The membrane integration, without the mechanisms.
        AdaptiveSoma : The mechanisms and their equations.
    """
    config: AdaptiveExponentialSomaConfig

    def __init__(self, config: AdaptiveExponentialSomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################