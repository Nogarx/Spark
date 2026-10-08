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
from math import prod
from spark.core.tracers import SaturableTracer
from spark.core.payloads import SpikeArray, FloatArray, SparkPayload
from spark.core.registry import register_interface, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.output.base import OutputInterface, OutputInterfaceConfig, OutputInterfaceOutput

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class TanhIntegratorConfig(OutputInterfaceConfig):
    """
        Configuration for `TanhIntegrator`.

        Parameters
        ----------
        saturation_freq : float, default 50.0
            Population firing rate at which an output reaches 1.0, in Hz.
        tau : float, default 5.0
            Decay constant of the integrator, in ms.
        smooth_trace : bool, default True
            Use a rise-and-decay trace instead of a single exponential, which removes the step
            each spike would otherwise leave in the output.
    """

    saturation_freq: float = dc.field(
        default = 50.0, 
        metadata = {
            'units': 'Hz',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Approximate average firing frequency at which the population needs to fire to sature the integrator.',
        })
    tau: float = dc.field(
        default = 5.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Decay time constant of the membrane potential of the units of the spiker.',
        })
    smooth: bool = dc.field(
        default = True, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Smooths the output signal with an using a double exponential moving average instead of a single EMA.',
        })
    smooth_tau: float = dc.field(
        default = 5.0, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Decay time constant for the smooth process.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class TanhIntegrator(OutputInterface):
    """
        Decodes spikes into a continuous signal.

        The input units are split into two groups. The spikes of each group are
        counted per step and passed through an exponential trace, so each output tracks the firing
        rate of its group.

        Parameters
        ----------
        config : TanhIntegratorConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        spikes : SpikeArray
            Spikes of the pool being read out.

        Output Ports
        ------------
        signal : FloatArray
            One value per group, of shape ``(num_outputs,)``. Reads 1.0 at ``saturation_freq``.

        Notes
        -----
        The trace is scaled so that a group firing at ``saturation_freq`` reads 1.0. Rates above
        it are not clipped and read above 1.0.
    """
    config: TanhIntegratorConfig

    def __init__(self, config: TanhIntegratorConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize internal variables
        self.saturation_freq = self.config.saturation_freq
        self.tau = self.config.tau
        self.smooth = self.config.smooth
        self.smooth_tau = self.config.smooth_tau

    def build(self, x_spikes: SpikeArray, y_spikes: SpikeArray) -> None:
        x_counts = prod(x_spikes.shape)
        y_counts = prod(y_spikes.shape)
        scale_factor = (1000/self.saturation_freq)/(self._dt*self.tau*jnp.array([x_counts, y_counts])) 
        # Initialize tracer
        self.trace = SaturableTracer(
            shape=(2,), 
            tau=self.tau, 
            scale=scale_factor, 
            dt=self._dt, 
            dtype=self._dtype
        )
        if self.smooth:
            self.smooth_trace = SaturableTracer(
                shape=(1,), 
                tau=self.smooth_tau, 
                scale=1/self.smooth_tau, 
                dt=self._dt, 
                dtype=self._dtype
            )

    def reset(self,) -> None:
        """
            Reset module to its default state.
        """
        self.trace.reset()

    def __call__(self, x_spikes: SpikeArray, y_spikes: SpikeArray) -> OutputInterfaceOutput:
        # Flat array
        """
            Counts the spikes of each group and integrates them.

            Parameters
            ----------
            spikes : SpikeArray
                Spikes of the pool being read out.

            Returns
            -------
            OutputInterfaceOutput
                Dictionary with one entry, ``signal``, of shape ``(num_outputs,)``.
        """
        x = jnp.sum(x_spikes.spikes.astype(self._dtype)).reshape((1,))
        y = jnp.sum(y_spikes.spikes.astype(self._dtype)).reshape((1,))
        # Count spikes in each output group.
        z = jnp.concatenate([x,y]).reshape(-1)
        # Integrate
        traces = self.trace(z)
        if self.smooth:
            diff = self.smooth_trace(traces[0] - traces[1])
        else:
            diff = traces[0] - traces[1]
        output = jnp.tanh(jnp.pi*diff)
        return {
            'signal': FloatArray(output)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################