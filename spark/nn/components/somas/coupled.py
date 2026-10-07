#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp

from spark.core.backend import Variable
from spark.core.decorators import spark_property
from spark.core.payloads import SparkPayload, CurrentArray
from spark.nn.components.base import ComponentConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class CoupledSomaConfig(ComponentConfig):
    """
        Configuration for `CoupledSoma`.

        Carries no field of its own. The coupling current is state written by another module,
        not a parameter.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class CoupledSoma:
    r"""
        Coupling of a soma to a dendritic compartment.

        Mixin adding a writable ``coupling_current`` property to a `Soma` subclass, the current
        another compartment injects into the membrane. It must precede the model in the base
        list::

            class CoupledLeakySoma(CoupledSoma, LeakySoma): ...

        Parameters
        ----------
        config : CoupledSomaConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        current : CurrentArray
            Current delivered to the membrane, in pA. The coupling current is added to it before
            the model sees it.
        inhibition_mask : BooleanMask, optional
            Marks the inhibitory units. Passed through to the model unchanged.

        Output Ports
        ------------
        spikes : SpikeArray
            Non-zero where the potential crossed the threshold.

        Properties
        ----------
        potential : PotentialArray
            Membrane potential, as held by the model this mixin extends. Read only.
        coupling_current : CurrentArray
            Current injected by the coupled compartment, in pA. Writable, so a `Neuron` can set
            it as an effect from the ``out_current`` port of a dendrite.

        Notes
        -----
        The mixin supplies one term of the soma step, the current offset:

        .. math::
            I_{\mathrm{eff}} = \Phi_{\mathrm{model}}(I + I_C)

        where :math:`I_C = g_C (V_d - V_s)` is what the dendrite reports. The current is added
        before the mechanisms of the model act on the input, so a refractory gate of
        `AdaptiveSoma` silences the dendritic drive along with the synaptic one, as the reset
        clamp of the reference model does.

        Within a `Neuron` the dendrite reads the ``potential`` property and the ``spikes``
        output of the soma, and the soma receives the dendritic current as an effect at the end
        of the step. The soma therefore integrates the current the dendrite computed on the
        previous step, one step of lag that keeps the wiring free of cycles and the execution
        order fixed. The dendrite sees the soma of the same step.

        See Also
        --------
        CoupledLeakySoma : Leaky membrane with the coupling.
        CoupledAdaptiveExponentialSoma : AdEx membrane with the coupling, the soma of the Ca-AdEx neuron.
        Dendrite : The compartment producing the coupling current.
    """
    config: CoupledSomaConfig

    def __init__(self, config: CoupledSomaConfig | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    def build(self, **abc_args: SparkPayload) -> None:
        super().build(**abc_args)
        # Current injected by the coupled compartment, written as an effect.
        self._coupling_current = Variable(jnp.zeros(self.units, dtype=self._dtype), dtype=self._dtype)

    @spark_property
    def coupling_current(self,) -> CurrentArray:
        return CurrentArray(self._coupling_current.value)

    @coupling_current.setter
    def coupling_current(self, new_current: CurrentArray) -> None:
        self._coupling_current.value = new_current.value.astype(self._dtype)

    def reset(self) -> None:
        """
            Resets component state.
        """
        super().reset()
        self._coupling_current.value = jnp.zeros(self.units, dtype=self._dtype)

    def _effective_current(self, current: jax.Array) -> jax.Array:
        """
            Adds the coupling current before the model acts on the input.
        """
        return super()._effective_current(current + self._coupling_current.value)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################