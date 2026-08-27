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
import typing as tp
import spark.core.utils as utils
from spark.core.payloads import SpikeArray, CurrentArray, FloatArray
from spark.core.backend import Variable
from spark.core.registry import register_module, register_config
from spark.core.config_validation import TypeValidator
from spark.nn.initializers.common import SparseUniformInitializerConfig
from spark.nn.components.synapses.base import Synapses, SynanpsesConfig
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class LinearSynapsesConfig(SynanpsesConfig):
    """
        Configuration for `LinearSynapses`.

        Parameters
        ----------
        units : tuple of int
            Shape of the postsynaptic pool.
        kernel : jax.Array or Initializer, default SparseUniformInitializerConfig()
            Synaptic weights, in pA. The kernel is built with shape ``units + input_shape``.
    """

    units: tuple[int, ...] = dc.field(
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'description': 'tuple[int, ...] of the postsynaptic pool of neurons.',
        })
    kernel: jax.Array | Initializer = dc.field(
        default_factory = SparseUniformInitializerConfig,
        metadata = {
            'units': 'pA',
            'validators': [
                TypeValidator,
            ], 
            'description': 'Synaptic weights initializer method. Note that we require the kernel entries to be in pA for numerical stability.',
        })
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class LinearSynapses(Synapses):
    r"""
        Linear synapse.

        The postsynaptic current is the weighted sum of the incoming spikes, with no extra dynamics,
        equivalent to delta increments in current.

        Parameters
        ----------
        config : LinearSynapsesConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        spikes : SpikeArray
            Presynaptic spikes.

        Output Ports
        ------------
        currents : CurrentArray
            Current delivered to each postsynaptic unit, of shape ``units``.

        Properties
        ----------
        kernel : FloatArray
            Synaptic weights, in pA, of shape ``units + input_shape``. Writable, which is how a
            plasticity rule updates them.

        Notes
        -----
        .. math::
            I_i = \sum_j W_{ij} s_j

        The kernel is built with shape ``units + input_shape`` and the sum runs over the
        presynaptic axes. Spikes marked asynchronous, as produced by `N2NDelays`, already carry
        one entry per (postsynaptic, presynaptic) pair; the postsynaptic axes are then matched
        elementwise and only the presynaptic axes are summed.

        References
        ----------
        .. [1] W. Gerstner, W. M. Kistler, R. Naud and L. Paninski, "Neuronal Dynamics: From
               Single Neurons to Networks and Models of Cognition", Chapter 1.3, Integrate-And-Fire
               Models. https://neuronaldynamics.epfl.ch/online/Ch1.S3.html

        See Also
        --------
        TracedSynapses : This model with an exponential postsynaptic current.
    """
    config: LinearSynapsesConfig

    def __init__(self, config: LinearSynapses | None = None, **kwargs):
        # Initialize super.
        super().__init__(config=config, **kwargs)
        # Initialize shapes
        self._output_shape = utils.validate_shape(self.config.units)
        

    def build(self, spikes: SpikeArray):
        # Initialize shapes
        self.async_spikes = spikes.async_spikes
        self._input_shape = utils.validate_shape(spikes.shape)
        self._real_input_shape = self._input_shape[len(self._output_shape):] if self.async_spikes else self._input_shape
        self._sum_axes = tuple(range(len(self._output_shape), len(self._output_shape)+len(self._real_input_shape)))
        # Initialize kernel
        kernel = self.config.init.kernel(
            init_kwargs = {'norm_axes': tuple(s for s in range(len(self._output_shape))),},
            key=self.get_rng_keys(1), shape=self._output_shape+self._real_input_shape, dtype=self._dtype,
        )
        self._kernel = Variable(kernel, dtype=self._dtype)
        
    def get_kernel(self,) -> FloatArray:
        return FloatArray(self._kernel.value)

    def get_flat_kernel(self,) -> FloatArray:
        return FloatArray(self._kernel.value.reshape(prod(self._output_shape), prod(self._real_input_shape)))

    def set_kernel(self, new_kernel: FloatArray) -> None:
        self._kernel.value = new_kernel.value

    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        return CurrentArray(jnp.sum(self._kernel.value * spikes.value, axis=self._sum_axes) )#* 1000.0)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################