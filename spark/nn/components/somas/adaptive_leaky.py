#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import InputSpec

import dataclasses as dc
from spark.core.tracers import Tracer
from spark.core.payloads import SpikeArray
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.somas.leaky import LeakySoma, LeakySomaConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class AdaptiveLeakySomaConfig(LeakySomaConfig):
    threshold_tau: float = dc.field(
        default = 20.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Adaptive action potential threshold decay constant.',
        })
    threshold_delta: float = dc.field(
        default = 100.0, 
        metadata = {
            'units': 'mV',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Adaptive action potential threshold after spike increment.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class AdaptiveLeakySoma(LeakySoma):
    """
        Adaptive leaky soma model.
    """
    config: AdaptiveLeakySomaConfig

    def __init__(self, config: AdaptiveLeakySomaConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    # NOTE: potential_rest is substracted to potential related terms to rebase potential at zero.
    def build(self, input_specs: dict[str, InputSpec]):
        super().build(input_specs)
        # Overwrite constant threshold with a tracer.
        self.threshold = Tracer(
            self._shape,
            tau=self.config.threshold_tau, 
            base=(self.config.threshold - self.config.potential_rest), 
            scale=self.config.threshold_delta, 
            dt=self._dt, dtype=self._dtype
        )

    def reset(self) -> None:
        """
            Resets component state.
        """
        super().reset()
        self.threshold.reset()

    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        # Compute spikes.
        spikes = super()._compute_spikes()
        # Update thresholds
        self.threshold(spikes.value)
        return spikes
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################