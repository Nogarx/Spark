#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import typing as tp
from spark.core.payloads import SpikeArray, IntegerArray
from spark.nn.components.base import Component, ComponentConfig
from spark.core.decorators import spark_property
from spark.core.backend import Variable

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class DelaysOutput(tp.TypedDict):
    """
        Output ports of a delay model.

        Attributes
        ----------
        out_spikes : SpikeArray
            Spikes emitted in the past, delivered on this step.
    """
    out_spikes: SpikeArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class DelaysConfig(ComponentConfig):
    """
        Base configuration for synaptic delay models.

        Carries no field of its own. Concrete models declare their own parameters.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=DelaysConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Delays(Component, tp.Generic[ConfigT]):
    """
        Base class for synaptic delay models.

        A delay model buffers incoming spikes and releases each one after the number of steps its
        kernel entry holds. Subclasses provide `_push`, which stores the spikes of the current
        step, and `_gather`, which reads the spikes due on it.

        Parameters
        ----------
        config : DelaysConfig
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        in_spikes : SpikeArray
            Spikes emitted on this step.

        Output Ports
        ------------
        out_spikes : SpikeArray
            Spikes emitted in the past and due on this step.

        Properties
        ----------
        kernel : IntegerArray
            Delay of every entry, in steps rather than in ms. Writable.

        Notes
        -----
        The delays are exposed as the writable ``kernel`` property, in steps rather than in ms.

        See Also
        --------
        NDelays : One delay per presynaptic unit.
        N2NDelays : One delay per (postsynaptic, presynaptic) pair.
    """
    _kernel: Variable

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)
    
    @spark_property
    def kernel(self,) -> IntegerArray:
        return IntegerArray(self._kernel.value)

    @kernel.setter
    def kernel(self, new_kernel: IntegerArray) -> None:
        self._kernel.value = new_kernel.value

    @abc.abstractmethod
    def _push(self, spikes: SpikeArray) -> None:
        """
            Push operation.
        """
        pass

    @abc.abstractmethod
    def _gather(self,) -> SpikeArray:
        """
            Gather operation.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
            Resets component state.
        """
        pass

    @abc.abstractmethod
    def __call__(self, in_spikes: SpikeArray) -> DelaysOutput:
        """
            Stores the incoming spikes and returns the ones due on this step.

            Parameters
            ----------
            in_spikes : SpikeArray
                Spikes emitted on this step.

            Returns
            -------
            DelaysOutput
                Dictionary with one entry, ``out_spikes``, the spikes whose delay elapsed on this
                step.
        """
        pass

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################