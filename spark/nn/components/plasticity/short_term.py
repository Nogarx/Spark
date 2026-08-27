#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING

from numpy import square
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs

import jax
import dataclasses as dc
import jax.numpy as jnp
from spark.core.tracers import Tracer
from spark.core.payloads import SpikeArray, FloatArray, CurrentArray
from spark.core.backend import Constant
from spark.core.registry import register_module, register_config
from spark.core.utils import get_einsum_dot_exp_string
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.components.plasticity.base import Plasticity, PlasticityConfig, PlasticityOutput
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class RUShortTermPlasticityConfig(PlasticityConfig):
    """
        Configuration for `RUShortTermPlasticity`.

        Parameters
        ----------
        r_tau : float or jax.Array or Initializer, default 750.0
            Recovery constant of the resource trace, in ms.
        u_tau : float or jax.Array or Initializer, default 50.0
            Decay constant of the usage trace, in ms.
        u_scale : float or jax.Array or Initializer, default 0.45
            Fraction of the available resources a spike consumes.
    """

    r_tau: float | jax.Array | Initializer = dc.field(
        default = 750.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the resource trace',
        })
    u_tau: float | jax.Array | Initializer = dc.field(
        default = 50.0, 
        metadata = {
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Time constant of the usage trace',
        })
    u_scale: float = dc.field(
        default = 0.45, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'Scale factor of the usage trace',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class RUShortTermPlasticity(Plasticity):
    r"""
        Resource-usage short term plasticity.

        Scales the synaptic current by the product of a usage variable and a resource variable.
        A spike raises usage and drains resources, and both relax back between spikes, so a burst
        is transmitted at a falling amplitude and recovers once the input pauses. Note that weights 
        are not touched, this is a gain applied to the current.

        Parameters
        ----------
        config : RUShortTermPlasticityConfig
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        pre_spikes : SpikeArray
            Presynaptic spikes, after any conduction delay.
        currents : CurrentArray
            Synaptic current before the short term gain is applied.

        Output Ports
        ------------
        kernel : CurrentArray
            The scaled current. The port is named ``kernel`` because every plasticity rule writes
            through that port, but this rule returns a current rather than weights.

        Notes
        -----
        With :math:`u` the usage trace, which rises by ``u_scale`` per spike and decays to zero,
        and :math:`r` the resource trace, which rests at one and is drained by :math:`u r` per
        spike, the transmitted current is

        .. math::
            I_{\mathrm{out}} = u \, r \, I_{\mathrm{in}}

        Facilitation and depression are set by the ratio of ``u_tau`` to ``r_tau``: a usage trace
        that outlives the resource trace facilitates, the reverse depresses.

        References
        ----------
        .. [1] M. Tsodyks, K. Pawelzik and H. Markram, "Neural Networks with Dynamic Synapses",
               Neural Computation 10(4), 821-835, 1998.
               https://doi.org/10.1162/089976698300017502
        .. [2] http://www.scholarpedia.org/article/Short-term_synaptic_plasticity
    """
    config: RUShortTermPlasticityConfig

    def __init__(self, config: RUShortTermPlasticityConfig | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, pre_spikes: SpikeArray, currents: CurrentArray) -> None:
        # Initialize shapes
        input_shape = pre_spikes.shape
        currents_shape = currents.shape

        # Initialize variables.
        # NOTE: Since variables used for plasticity may have very different shapes and initialization patterns
        # variables should almost always be initialized using self._initialize_variable. This method automatically
        # handles the most common initialization patterns and tries to keep the variable the smallest shape possible
        # that is still compute efficient for plasticity computation (THIS APPRAOCH IS NOT ALWAYS MEMORY EFFICIENT).
        _initialize_variable_fn = lambda var: self._initialize_variable(var, shape=currents_shape, dtype=currents.dtype)
        _r_tau = _initialize_variable_fn(self.config.init.r_tau)
        _u_tau = _initialize_variable_fn(self.config.init.u_tau)
        _u_scale = _initialize_variable_fn(self.config.init.u_scale)
        # Tracers.
        self.r_tracer = Tracer(currents_shape, tau=_r_tau, scale=-1.0, base=1.0)
        self.u_tracer = Tracer(currents_shape, tau=_u_tau, scale=_u_scale, base=0.0)
        async_spikes =pre_spikes.async_spikes
        self._pre_reshape = input_shape if async_spikes else (len(currents_shape) - len(input_shape)) * (1,) + input_shape
            
    def reset(self) -> None:
        """
            Resets component state.
        """
        self.r_tracer.reset()
        self.u_tracer.reset()

    def _compute_current_update(self, pre_spikes: SpikeArray, currents: CurrentArray) -> jax.Array:
        """
            Computes next kernel update.
        """
        # Squeeze
        _pre_spikes = pre_spikes.spikes.reshape(self._pre_reshape)
        _currents = currents.value
        # Update and get current trace value
        # Update usage
        u_trace = self.u_tracer(_pre_spikes)
        # Compute RU
        trace_RU = u_trace * self.r_tracer.value
        # Update resources
        self.r_tracer(u_trace * self.r_tracer.value * _pre_spikes)
        return _currents * trace_RU
        
    def __call__(self, pre_spikes: SpikeArray, currents: CurrentArray) -> PlasticityOutput:
        """
            Applies the short term gain to the synaptic current.

            Parameters
            ----------
            pre_spikes : SpikeArray
                Presynaptic spikes, after any conduction delay.
            currents : CurrentArray
                Synaptic current before the gain is applied.

            Returns
            -------
            PlasticityOutput
                Dictionary with one entry, ``kernel``, carrying the scaled `CurrentArray`.
        """
        return {
            'kernel': CurrentArray(self._compute_current_update(pre_spikes, currents))
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################