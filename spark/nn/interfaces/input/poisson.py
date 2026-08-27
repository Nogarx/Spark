#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs

import jax
import dataclasses as dc
import spark.core.utils as utils
from spark.core.payloads import SpikeArray, FloatArray
from spark.core.registry import register_interface, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.nn.interfaces.input.base import InputInterface, InputInterfaceConfig, InputInterfaceOutput

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class PoissonSpikerConfig(InputInterfaceConfig):
    """
        Configuration for `PoissonSpiker`.

        Parameters
        ----------
        max_freq : float, default 100.0
            Firing rate reached by an input of 1.0, in Hz.
    """

    max_freq: float = dc.field(
        default = 100.0, 
        metadata = {
            'units': 'Hz',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Maximum firing frequency of the spiker.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_interface
class PoissonSpiker(InputInterface):
    r"""
        Stochastic rate encoding of a continuous signal.

        Each unit emits a spike on a step with a probability proportional to its input, drawn
        independently every step. Repeated presentations of the same input give different spike
        trains with the same expected rate.

        Parameters
        ----------
        config : PoissonSpikerConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        signal : FloatArray
            Value to encode, expected in ``[0, 1]``.

        Output Ports
        ------------
        spikes : SpikeArray
            One spike train per input, of the same shape as ``signal``.

        Notes
        -----
        .. math::
            P(s_i = 1) = \frac{f_{\max} \Delta t}{1000} x_i

        with ``dt`` in ms. Inputs above 1.0 saturate at one spike per step, and the encoding is
        only linear while :math:`f_{\max} \Delta t / 1000 \le 1`.

        See Also
        --------
        LinearSpiker : Deterministic encoding of the same signal.
    """
    config: PoissonSpikerConfig

    def __init__(self, config: PoissonSpikerConfig | None = None, **kwargs):
        # Initialize super
        super().__init__(config=config, **kwargs)
        # Initialize variables
        self.max_freq = self.config.max_freq
        self._scale = self._dt * (self.max_freq / 1000)

    def build(self, signal: FloatArray) -> None:
        # Initialize shapes
        self._shape = utils.validate_shape(signal.shape)

    def __call__(self, signal: FloatArray) -> InputInterfaceOutput:
        """
            Encodes the signal as spikes.

            Parameters
            ----------
            signal : FloatArray
                Value to encode, expected in ``[0, 1]``.

            Returns
            -------
            InputInterfaceOutput
                Dictionary with one entry, ``spikes``, of the same shape as ``signal``.
        """
        spikes = (jax.random.uniform(self.get_rng_keys(1), shape=self._shape) < self._scale * signal.value).astype(self._dtype)
        return {
            'spikes': SpikeArray(spikes)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################