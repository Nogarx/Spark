#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.specs import PortSpecs

import abc
import jax
import jax.numpy as jnp
import typing as tp
import spark.core.utils as utils
from spark.core.backend import Variable
from spark.nn.components.base import Component, ComponentConfig
from spark.core.payloads import SpikeArray, CurrentArray, PotentialArray, BooleanMask
from spark.core.decorators import spark_property

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class SomaOutput(tp.TypedDict):
    """
        Output ports of a soma model.

        Attributes
        ----------
        spikes : SpikeArray
            One entry per unit, non-zero where the membrane potential crossed the threshold.
    """
    spikes: SpikeArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SomaConfig(ComponentConfig):
    """
        Base configuration for soma models.

        Carries no field of its own. Concrete models declare their own parameters.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=SomaConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Soma(Component, tp.Generic[ConfigT]):
    """
        Base class for soma models.

        Base component representing membrane potential dynamics. Soma models subclass this component
        in order to implement specific some dynamics. Standard auxiliary methods are provided by this
        component.

            I_eff = _effective_current(I)
            V'    = _integrate(V, I_eff)
            V''   = _post_integrate(V')
            s     = (V'' > _effective_threshold()) and _spike_mask()
            V     = s ? V_reset : V''
                    _after_spike(s)

        Only `_integrate` is required. The other hooks return their argument unchanged, so a
        model that is a membrane integration and nothing else defines one method.

        Parameters
        ----------
        config : SomaConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        current : CurrentArray
            Current delivered to the membrane, in pA.
        inhibition_mask : BooleanMask, optional
            Marks the inhibitory units. Supplied by the enclosing `Neuron`. It is stamped onto the
            emitted spikes and does not affect the integration.

        Output Ports
        ------------
        spikes : SpikeArray
            Non-zero where the potential crossed the threshold on this step.

        Properties
        ----------
        potential : PotentialArray
            Membrane potential, relative to ``potential_rest``. Read only.

        Notes
        -----
        A subclass owns a ``threshold`` and a ``potential_reset`` constant. A mechanism written
        against those, rather than against a particular integration, composes with every model in
        this package.

        Two kinds of subclass use the hooks. A model whose own state is integrated alongside the
        membrane, such as the recovery variable of `IzhikevichSoma`, updates that state in
        `_post_integrate` and `_after_spike`. `AdaptiveSoma` wraps an existing model by driving
        the current, the potential, the threshold and the spike veto.

        See Also
        --------
        AdaptiveSoma : Refractoriness, potential clamp, threshold and current adaptation.
        LeakySoma : Leaky integrate-and-fire membrane.
        ExponentialSoma : Exponential integrate-and-fire membrane.
        IzhikevichSoma : Quadratic membrane with a recovery variable.
    """
    config: ConfigT

    def __init__(self, config: ConfigT | None = None, **kwargs):
        # Initialize super.
        super().__init__(config = config, **kwargs)

    def build(self, current: CurrentArray, inhibition_mask: BooleanMask | None = None) -> None:
        # Initialize shapes
        self.units = utils.validate_shape(current.shape)
        # Initialize variables
        self._potential = Variable(jnp.zeros(self.units, dtype=self._dtype), dtype=self._dtype)

    @spark_property
    def potential(self,) -> PotentialArray:
        return PotentialArray(self._potential.value)

    def reset(self) -> None:
        """
            Resets neuron states to their initial values.
        """
        self._potential.value = jnp.zeros(self.units, dtype=self._dtype)

    @abc.abstractmethod
    def _integrate(self, potential: jax.Array, current: jax.Array) -> jax.Array:
        """
            Membrane integration. The only part of the step every model has to provide.
        """
        pass

    def _effective_current(self, current: jax.Array) -> jax.Array:
        """
            The current the membrane actually integrates.
        """
        return current

    def _post_integrate(self, potential: jax.Array) -> jax.Array:
        """
            The membrane potential as it stands after integration and before the threshold is
            tested. State that follows the potential within the step is updated here.
        """
        return potential

    def _effective_threshold(self) -> jax.Array:
        """
            The value the membrane potential is tested against.
        """
        return self.threshold.value

    def _spike_mask(self) -> jax.Array | None:
        """
            Veto applied to the spikes, or None when nothing vetoes them.
        """
        return None

    def _after_spike(self, spikes: jax.Array) -> None:
        """
            Called once the spikes are out and the reset has been applied, so the membrane
            potential reads as it will at the start of the next step.
        """
        pass

    def _fire_and_reset(
            self,
            potential: jax.Array,
            threshold: jax.Array,
            spike_mask: jax.Array | None = None,
        ) -> jax.Array:
        """
            Emits the spikes and applies the after spike reset. Shared by every soma model.
        """
        spikes = jnp.greater(potential, threshold)
        if spike_mask is not None:
            spikes = jnp.logical_and(spikes, spike_mask)
        spikes = spikes.astype(self._dtype)
        self._potential.value = spikes * self.potential_reset.value + (1 - spikes) * potential
        return spikes

    def __call__(
            self,
            current: CurrentArray,
            inhibition_mask: BooleanMask | bool | None = None,
        ) -> SomaOutput:
        """
            Advances the membrane one step and emits the spikes.

            Parameters
            ----------
            current : CurrentArray
                Current delivered to the membrane, in pA.
            inhibition_mask : BooleanMask or bool, optional
                Marks the inhibitory units. Stamped onto the emitted spikes without affecting the
                integration.

            Returns
            -------
            SomaOutput
                Dictionary with one entry, ``spikes``, non-zero where the potential crossed the
                threshold.
        """
        potential = self._integrate(self._potential.value, self._effective_current(current.value))
        potential = self._post_integrate(potential)
        spikes = self._fire_and_reset(potential, self._effective_threshold(), self._spike_mask())
        self._after_spike(spikes)
        return {
            'spikes': SpikeArray(spikes=spikes, inhibition_mask=inhibition_mask)
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
