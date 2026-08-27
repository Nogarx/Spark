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
        IzhikevichSoma model configuration class.
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
    """
        Izhikevich soma model.

        The recovery variable is integrated together with the membrane potential and reads the
        potential in the middle of the step, so it is part of the model rather than something
        attached to it: it is updated in _post_integrate and _after_spike. That places nothing
        in the way of the adaptation extension, and AdaptiveIzhikevichSoma is this model with
        refractoriness, the potential clamp and threshold adaptation available to it.

        Unlike the current based somas this model is not rebased at zero, since the quadratic
        term is not translation invariant and rebasing would only cost an extra addition per
        term.

        Init:
            units: tuple[int, ...]
            potential_rest: float | jax.Array
            potential_reset: float | jax.Array
            resistance: float | jax.Array
            threshold: float | jax.Array
            recovery_timescale: float | jax.Array
            recovery_sensitivity: float | jax.Array
            recovery_update: float | jax.Array

        Input:
            current: CurrentArray

        Output:
            spikes: SpikeArray

        Reference:
            Simple Model of Spiking Neurons
            Eugene M. Izhikevich
            IEEE Transactions on Neural Networks, vol. 14, no. 6, pp. 1569-1572, Nov. 2003
            https://doi.org/10.1109/TNN.2003.820440
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
        AdaptiveIzhikevichSoma model configuration class.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class AdaptiveIzhikevichSoma(AdaptiveSoma, IzhikevichSoma):
    """
        Izhikevich soma model with the adaptation extension.

        Input:
            current: CurrentArray

        Output:
            spikes: SpikeArray
    """
    config: AdaptiveIzhikevichSomaConfig

    def __init__(self, config: AdaptiveIzhikevichSomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################