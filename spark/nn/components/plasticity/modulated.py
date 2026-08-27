#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs

import jax
from spark.core.payloads import SpikeArray, FloatArray
from spark.nn.components.plasticity.base import Plasticity, PlasticityConfig, PlasticityOutput

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ModulatedPlasticityConfig(PlasticityConfig):
    """
        Configuration for `ModulatedPlasticity`.

        Carries no field of its own. The third factor arrives on a port rather than through the
        configuration.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ModulatedPlasticity:
    r"""
        Third factor for plasticity rules.

        Mixin adding a ``modulation`` input port to a `Plasticity` subclass and scaling the
        weight change by whatever arrives on it. It must precede the rule in the base list::

            class ModulatedHebbianRule(ModulatedPlasticity, HebbianRule): ...

        Parameters
        ----------
        config : ModulatedPlasticityConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        modulation : FloatArray
            Third factor scaling the whole update.
        pre_spikes : SpikeArray
            Presynaptic spikes, after any conduction delay.
        post_spikes : SpikeArray
            Postsynaptic spikes emitted on this step.
        kernel : FloatArray
            Current synaptic weights, read from the synapse.

        Output Ports
        ------------
        kernel : FloatArray
            The updated weights, written back onto the synapse as an effect.

        Notes
        -----
        The rule supplies its terms through `Plasticity._kernel_delta`, and the mixin folds the
        third factor into the learning rate rather than into the result:

        .. math::
            W \leftarrow \mathrm{clip}\left(
                W + \Delta t \, (\eta M) \, \Delta W \right)

        A modulation of zero freezes the weights, and a negative modulation reverses the sign of
        the update. The bounds are those of the rule, so an upper bound declared by the rule is
        still applied.

        Scaling the rate rather than the result is what keeps the arithmetic associated the same
        way as in an unmodulated rule, which matters in half precision.

        The signal is expected in a shape that broadcasts against the kernel: a scalar, the
        presynaptic shape, the postsynaptic shape, or the kernel shape itself.

        The port is added by overriding ``__call__``, whose signature is what the framework reads
        the input ports from. `build` drops the extra argument before handing the rest to the
        rule.

        See Also
        --------
        Plasticity : The base the rule derives from, and the hooks it fills in.
        ModulatedHebbianRule : `HebbianRule` with this mixin.
        ModulatedQuadrupletRule : `QuadrupletRule` with this mixin.
    """
    config: ModulatedPlasticityConfig

    def __init__(self, config: ModulatedPlasticityConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(
            self,
            modulation: FloatArray,
            pre_spikes: SpikeArray,
            post_spikes: SpikeArray,
            kernel: FloatArray,
        ) -> None:
        # The third factor carries no state, so the rule builds against its own ports alone.
        super().build(pre_spikes=pre_spikes, post_spikes=post_spikes, kernel=kernel)

    def __call__(
            self,
            modulation: FloatArray,
            pre_spikes: SpikeArray,
            post_spikes: SpikeArray,
            kernel: FloatArray,
        ) -> PlasticityOutput:
        """
            Computes the weights for the next step, scaled by the third factor.

            Parameters
            ----------
            modulation : FloatArray
                Third factor scaling the whole update.
            pre_spikes : SpikeArray
                Presynaptic spikes, after any conduction delay.
            post_spikes : SpikeArray
                Postsynaptic spikes emitted on this step.
            kernel : FloatArray
                Current synaptic weights.

            Returns
            -------
            PlasticityOutput
                Dictionary with one entry, ``kernel``, the updated weights.
        """
        delta = self._kernel_delta(pre_spikes, post_spikes, kernel)
        return self._apply_delta(kernel, (self._learning_rate() * modulation.value) * delta)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
