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
from spark.core.tracers import Tracer, RDTracer
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.backend import Variable
from spark.core.registry import register_interface, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.output.base import OutputInterface, OutputInterfaceConfig, OutputInterfaceOutput

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ExponentialIntegratorConfig(OutputInterfaceConfig):
    """
        Configuration for `ExponentialIntegrator`.

        Parameters
        ----------
        num_outputs : int
            Number of output signals. The input units are split equally among them.
        saturation_freq : float, default 50.0
            Population firing rate at which an output reaches 1.0, in Hz.
        tau : float, default 5.0
            Decay constant of the integrator, in ms.
        output_map : jax.Array or None, default None
            Index of the output each input unit contributes to. Must have one entry per input
            unit. Assigned automatically when None.
        shuffle : bool, default True
            Draw the automatic assignment at random rather than in order. Only read when
            ``output_map`` is None.
        smooth_trace : bool, default True
            Use a rise-and-decay trace instead of a single exponential, which removes the step
            each spike would otherwise leave in the output.
    """

    num_outputs: int = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Number of output signals. Input spikes are distributed equally among the output signals. \
                            If num_outputs does not exactly divide the number of incomming spikes then an approximately \
                            even assigment is used.',
        })
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
    output_map: jax.Array | None = dc.field(
        default = None, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Integer array that assigns each input to each output integrator. Must have the same (flatten) shape as the input',
        })
    shuffle: bool = dc.field(
        default = True, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Shuffles the input spikes, otherwise they are used sequentially to create the output signal.',
        })
    smooth_trace: bool = dc.field(
        default = True, 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Smooths the output signal with an using a double exponential moving average instead of a single EMA.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class ExponentialIntegrator(OutputInterface):
    """
        Decodes spikes into a continuous signal.

        The input units are split into ``num_outputs`` groups. The spikes of each group are
        counted per step and passed through an exponential trace, so each output tracks the firing
        rate of its group.

        Parameters
        ----------
        config : ExponentialIntegratorConfig, optional
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

        Assignment is by ``output_map`` when given. Otherwise the units are dealt out evenly,
        shuffled when ``shuffle`` is set, which spreads each output over the pool rather than
        over one contiguous block of it.
    """
    config: ExponentialIntegratorConfig

    def __init__(self, config: ExponentialIntegratorConfig = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize internal variables
        self.num_outputs = self.config.num_outputs
        self.saturation_freq = self.config.saturation_freq
        self.tau = self.config.tau
        self.shuffle = self.config.shuffle
        self.smooth_trace = self.config.smooth_trace

    def build(self, spikes: SpikeArray) -> None:
        # Output mapping.
        in_dim = prod(spikes.shape)
        out_dim = self.num_outputs
        base = in_dim // out_dim
        remainder = in_dim % out_dim
        counts = jnp.concatenate([
            jnp.full(remainder, base + 1, dtype=jnp.int32),
            jnp.full(out_dim - remainder, base, dtype=jnp.int32)
        ])
        if self.config.output_map is not None:
            output_map = self.config.output_map
        else:
            output_map = jnp.repeat(jnp.arange(out_dim, dtype=jnp.int32), counts)
            if not self.shuffle:
                output_map = output_map
            else: 
                output_map = jax.random.permutation(self.get_rng_keys(1), output_map)
        self._indices = Variable(output_map, dtype=jnp.uint32)

        # Initialize tracer
        if self.smooth_trace:
            self.trace = RDTracer(
                shape=self.num_outputs, 
                tau_rise=self.tau, 
                tau_decay=10.0,
                scale_rise=(1000/self.saturation_freq)/(self._dt*self.tau*counts), 
                scale_decay=1/10,
                dt=self._dt, 
                dtype=self._dtype
            )
        else:
            self.trace = Tracer(
                shape=self.num_outputs, 
                tau=self.tau, 
                scale=(1000/self.saturation_freq)/(self._dt*self.tau*counts), 
                dt=self._dt, 
                dtype=self._dtype
            )

    def reset(self,):
        """
            Reset module to its default state.
        """
        self.trace.reset()

    def __call__(self, spikes: SpikeArray) -> OutputInterfaceOutput:
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
        x = spikes.spikes.reshape(-1).astype(self._dtype)
        # Count spikes in each output group.
        x = jax.ops.segment_sum(
            x, 
            self._indices, 
            indices_are_sorted=(not self.shuffle), 
            num_segments=self.num_outputs
        )
        # Integrate
        output = self.trace(x)
        return {
            'signal': FloatArray(output)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################