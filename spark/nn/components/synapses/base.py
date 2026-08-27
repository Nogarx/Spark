#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import typing as tp
from spark.nn.components.base import Component, ComponentConfig
from spark.core.backend import Variable
from spark.core.payloads import SpikeArray, CurrentArray, FloatArray
from spark.core.decorators import spark_property

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class SynanpsesOutput(tp.TypedDict):
    """
        Output ports of a synapse model.

        Attributes
        ----------
        currents : CurrentArray
            Current delivered to each postsynaptic unit.
    """
    currents: CurrentArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SynanpsesConfig(ComponentConfig):
    """
        Base configuration for synapse models.

        Carries no field of its own. Concrete models declare their own parameters.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=SynanpsesConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Synapses(Component, tp.Generic[ConfigT]):
    """
        Base class for synapse models.

        A synapse model turns presynaptic spikes into postsynaptic current. Subclasses provide
        `_dot`, which is the whole of the step.

        Parameters
        ----------
        config : SynanpsesConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        spikes : SpikeArray
            Presynaptic spikes.

        Output Ports
        ------------
        currents : CurrentArray
            Current delivered to each postsynaptic unit.

        Properties
        ----------
        kernel : FloatArray
            Synaptic weights, in pA. Writable, which is how a plasticity rule updates them.

        Notes
        -----
        Kernel entries are in pA. The framework runs in half precision by default, and nA-scale
        weights lose too much of the mantissa to be summed reliably.

        The weights are exposed as the writable ``kernel`` property, which is what lets a
        plasticity rule read them and write them back.

        See Also
        --------
        LinearSynapses : Weighted sum of the incoming spikes.
        TracedSynapses : Weighted sum filtered by a single exponential.
    """
    _kernel: Variable

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)

    @spark_property
    def kernel(self,) -> FloatArray:
        return FloatArray(self._kernel.value)

    @kernel.setter
    def kernel(self, new_kernel: FloatArray) -> None:
        self._kernel.value = new_kernel.value

    @abc.abstractmethod
    def get_kernel(self,) -> FloatArray:
        pass

    @abc.abstractmethod
    def set_kernel(self, new_kernel: FloatArray) -> None:
        pass

    @abc.abstractmethod
    def _dot(self, spikes: SpikeArray) -> CurrentArray:
        pass

    def __call__(self, spikes: SpikeArray) -> SynanpsesOutput:
        """
            Converts presynaptic spikes into postsynaptic current.

            Parameters
            ----------
            spikes : SpikeArray
                Presynaptic spikes.

            Returns
            -------
            SynanpsesOutput
                Dictionary with one entry, ``currents``, the current delivered to each postsynaptic
                unit.
        """
        return {
            'currents': self._dot(spikes)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################