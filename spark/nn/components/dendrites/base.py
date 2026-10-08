#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax
import jax.numpy as jnp
import typing as tp
import dataclasses as dc

import spark.core.utils as utils
from spark.core.backend import Variable, Constant
from spark.core.decorators import spark_property
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.core.payloads import CurrentArray, PotentialArray, SpikeArray, FloatArray
from spark.nn.components.base import Component, ComponentConfig
from spark.nn.initializers.base import Initializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class DendriteOutput(tp.TypedDict):
    """
        Output ports of a dendrite model.

        Attributes
        ----------
        out_current : CurrentArray
            One entry per unit, the axial current the dendrite delivers to the soma, in pA.
            Positive when the dendrite is more depolarized than the soma.
        plateau : FloatArray
            One entry per unit, how far the dendrite is into a plateau. Rises from 0 to 1 as
            the membrane crosses ``threshold``, and reads 0.5 exactly at it.
    """
    out_current: CurrentArray
    plateau: FloatArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class DendriteConfig(ComponentConfig):
    """
        Base configuration for dendrite models.

        Holds what every dendrite in this package shares: its rest potential, the axial coupling
        to the soma, the back-propagating action potential and the plateau reading. Concrete
        models add the parameters of their membrane.

        Parameters
        ----------
        potential_rest : float or jax.Array or Initializer, default -55.0
            Dendritic leak reversal potential, in mV. Potentials are stored relative to it.
        coupling : float or jax.Array or Initializer, default 19.777
            Axial conductance between the soma and the dendrite, in nS. The dendrite receives
            ``coupling * (V_soma - V_dendrite)`` and delivers the opposite current to the soma.
        soma_potential_rest : float or jax.Array or Initializer or None, default None
            Rest potential of the soma the dendrite is coupled to, in mV. A soma reports its
            potential relative to its own rest, so this is needed to place both compartments on
            one scale. ``None`` reads as ``potential_rest``, i.e. no offset.
        bap_conductance : float or jax.Array or Initializer, default 27.996
            Peak conductance reached by one back-propagating action potential, in nS.
        bap_reversal : float or jax.Array or Initializer, default 0.0
            Reversal potential of the back-propagating conductance, in mV.
        bap_rise : float or jax.Array or Initializer, default 0.2
            Rise constant of the back-propagating conductance, in ms. Must be shorter than
            ``bap_decay``.
        bap_decay : float or jax.Array or Initializer, default 3.0
            Decay constant of the back-propagating conductance, in ms.
        threshold : float or jax.Array or Initializer, default -25.0
            Potential above which the dendrite counts as being in a plateau, in mV.
        plateau_slope : float or jax.Array or Initializer, default 2.0
            Slope of the plateau reading, in mV. Smaller values approach a hard threshold.

        Notes
        -----
        The defaults are the distal compartment of the Ca-AdEx neuron of Pastorelli et al.
        (2025), Supplementary Table S1. The table lists ``w_BAP`` in mV; the reference
        implementation uses it as the peak of a conductance, in nS, and so does this package.
    """

    potential_rest: float | jax.Array | Initializer = dc.field(
        default = -55.0,
        metadata = {
            'units': 'mV',
            'validators': [TypeValidator],
            'description': 'Dendrite leak reversal potential.',
        })
    coupling: float | jax.Array | Initializer = dc.field(
        default = 5.0,
        metadata = {
            'units': 'nS', # 1/GΩ
            'validators': [TypeValidator],
            'description': 'Soma-dendrite axial coupling conductance.',
        })
    soma_potential_rest: float | jax.Array | Initializer | None = dc.field(
        default = None,
        metadata = {
            'units': 'mV',
            'validators': [TypeValidator],
            'description': 'Rest potential of the coupled soma. None reads as potential_rest.',
        })
    bap_conductance: float | jax.Array | Initializer = dc.field(
        default = 28.0,
        metadata = {
            'units': 'nS',
            'validators': [TypeValidator],
            'description': 'Peak conductance of one back-propagating action potential.',
        })
    bap_reversal: float | jax.Array | Initializer = dc.field(
        default = 0.0,
        metadata = {
            'units': 'mV',
            'validators': [TypeValidator],
            'description': 'Reversal potential of the back-propagating conductance.',
        })
    bap_rise: float | jax.Array | Initializer = dc.field(
        default = 0.2,
        metadata = {
            'units': 'ms',
            'validators': [TypeValidator, PositiveValidator],
            'description': 'Rise constant of the back-propagating conductance.',
        })
    bap_decay: float | jax.Array | Initializer = dc.field(
        default = 3.0,
        metadata = {
            'units': 'ms',
            'validators': [TypeValidator, PositiveValidator],
            'description': 'Decay constant of the back-propagating conductance.',
        })
    threshold: float | jax.Array | Initializer = dc.field(
        default = -25.0,
        metadata = {
            'units': 'mV',
            'validators': [TypeValidator],
            'description': 'Plateau detection threshold.',
        })
    plateau_slope: float | jax.Array | Initializer = dc.field(
        default = 2.0,
        metadata = {
            'units': 'mV',
            'validators': [TypeValidator, PositiveValidator],
            'description': 'Slope of the plateau reading.',
        })

ConfigT = tp.TypeVar("ConfigT", bound=DendriteConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Dendrite(Component, tp.Generic[ConfigT]):
    """
        Base class for dendrite models.

        A dendrite owns a membrane potential, exchanges an axial current with the soma it is
        attached to, and receives the somatic spikes as a back-propagating action potential. The
        step is fixed here and a subclass fills in the membrane::

            V_s    = soma_potential + (E_soma - E_dendrite)
            g_BAP  = _bap_conductance(spikes)
            I_eff  = _effective_current(I)
            V'     = _integrate(V, I_eff, V_s, g_BAP)
                     _update_state(V, V')
            p      = _plateau_computation(V', _effective_threshold())
            I_out  = coupling * (V' - V_s)

        Only `_integrate` is required. The other hooks return their argument unchanged or do
        nothing, so a model that is a membrane integration and nothing else defines one method.

        Parameters
        ----------
        config : DendriteConfig, optional
            Model configuration. Its fields may also be given as keyword arguments.

        Input Ports
        -----------
        in_current : CurrentArray
            Current delivered to the dendrite, in pA. Synaptic input and any injected current.
        soma_potential : PotentialArray, optional
            Potential of the soma the dendrite is coupled to, relative to the soma's rest. Wire
            the ``potential`` property of the soma onto it. Without it the dendrite is an
            isolated compartment: no axial current flows in either direction.
        spikes : SpikeArray, optional
            Somatic spikes. Each one triggers a back-propagating action potential. Without it
            no back-propagation takes place.
        injected_current : CurrentArray, optional
            Current injected into the dendrite besides the synaptic one, in pA, summed with
            ``in_current``. Meant for instructive or teaching currents that are not synaptic.

        Output Ports
        ------------
        out_current : CurrentArray
            Axial current delivered to the soma, in pA. Inside a `Neuron` it is written onto the
            ``coupling_current`` property of a `CoupledSoma` as an effect, which closes the loop
            between the two compartments without a cycle in the wiring.
        plateau : FloatArray
            How far the dendrite is into a plateau, in ``[0, 1]``.

        Properties
        ----------
        potential : PotentialArray
            Dendritic membrane potential, relative to ``potential_rest``. Read only.

        Notes
        -----
        The back-propagating action potential follows the reference implementation of the
        Ca-AdEx neuron: a conductance with a double exponential time course, normalized so that
        one spike peaks at ``bap_conductance``, driving the membrane towards ``bap_reversal``.

        There is no conduction delay parameter. The spike takes on the order of a millisecond
        to travel from the soma to the calcium hot zone, which is at or below the step sizes
        Spark is typically run at, and the reference fit put its delay at the smallest value
        its simulator allowed, one 0.1 ms step. A somatic spike therefore enters the kernel on
        the step it arrives and, since the kernel starts from zero, its conductance acts from
        the next step on. To model a longer conduction delay, put an `NDelays` between the
        ``spikes`` output of the soma and the ``spikes`` port of the dendrite.

        Potentials are stored relative to ``potential_rest``. The soma potential arrives
        relative to the soma's own rest and is shifted by ``soma_potential_rest -
        potential_rest`` before use, so ``soma_potential_rest`` has to name the rest potential
        of the soma actually wired in.

        The plateau is reported as a smooth reading of the threshold crossing rather than as a
        flag, so it can drive a trace or gate a plasticity rule directly. It reads 0.5 at
        ``threshold``, so comparing it against 0.5 recovers the hard flag.

        See Also
        --------
        LeakyDendrite : Passive membrane.
        CalciumDendrite : Calcium hot zone with a regenerative calcium spike.
        CoupledSoma : Soma mechanism receiving ``out_current``.
    """
    config: ConfigT

    def __init__(self, config: ConfigT | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config=config, **kwargs)

    # NOTE: potential_rest is substracted to potential related terms to rebase potential at zero.
    def build(
            self,
            in_current: CurrentArray,
            soma_potential: PotentialArray | None = None,
            spikes: SpikeArray | None = None,
            injected_current: CurrentArray | None = None,
        ) -> None:
        # Initialize shapes
        self.units = utils.validate_shape(in_current.shape)
        # Initialize variables.
        init = self.config.init
        _draw = lambda name: getattr(init, name)(key=self.get_rng_keys(1), shape=self.units, dtype=self._dtype)
        _potential_rest = _draw('potential_rest')
        _soma_potential_rest = _potential_rest if self.config.soma_potential_rest is None else _draw('soma_potential_rest')
        _coupling = _draw('coupling')
        _bap_conductance = _draw('bap_conductance')
        _bap_reversal = _draw('bap_reversal')
        _bap_rise = jnp.asarray(_draw('bap_rise'), dtype=jnp.float32)
        _bap_decay = jnp.asarray(_draw('bap_decay'), dtype=jnp.float32)
        _threshold = _draw('threshold')
        _plateau_slope = _draw('plateau_slope')
        if bool(jnp.any(_bap_rise >= _bap_decay)):
            raise ValueError(
                f'Module "{self.name}": "bap_rise" must be shorter than "bap_decay".'
            )
        # Membrane. Substract potential_rest to potential related terms to rebase potential at zero.
        self.potential_rest = Constant(_potential_rest, dtype=self._dtype)
        self.soma_offset = Constant(_soma_potential_rest - _potential_rest, dtype=self._dtype)
        self.coupling = Constant(_coupling, dtype=self._dtype)
        # Back-propagating action potential. Double exponential conductance, one spike peaks at
        # bap_conductance.
        peak_time = (_bap_rise * _bap_decay) / (_bap_decay - _bap_rise) * jnp.log(_bap_decay / _bap_rise)
        peak = jnp.exp(-peak_time / _bap_decay) - jnp.exp(-peak_time / _bap_rise)
        self.bap_conductance = Constant(_bap_conductance / peak, dtype=self._dtype)
        self.bap_reversal = Constant(_bap_reversal - _potential_rest, dtype=self._dtype)
        self.bap_rise_rate = Constant(-jnp.expm1(-self._dt / _bap_rise), dtype=self._dtype)
        self.bap_fall_rate = Constant(-jnp.expm1(-self._dt / _bap_decay), dtype=self._dtype)
        # Plateau reading.
        self.threshold = Constant(_threshold - _potential_rest, dtype=self._dtype)
        self.plateau_slope = Constant(_plateau_slope, dtype=self._dtype)
        # State.
        self._potential = Variable(jnp.zeros(self.units, dtype=self._dtype), dtype=self._dtype)
        self._bap_rise = Variable(jnp.zeros(self.units, dtype=self._dtype), dtype=self._dtype)
        self._bap_fall = Variable(jnp.zeros(self.units, dtype=self._dtype), dtype=self._dtype)

    @spark_property
    def potential(self,) -> PotentialArray:
        return PotentialArray(self._potential.value)

    def reset(self) -> None:
        """
            Resets the dendrite state to its initial value.
        """
        self._potential.value = jnp.zeros(self.units, dtype=self._dtype)
        self._bap_rise.value = jnp.zeros(self.units, dtype=self._dtype)
        self._bap_fall.value = jnp.zeros(self.units, dtype=self._dtype)

    @abc.abstractmethod
    def _integrate(
            self,
            potential: jax.Array,
            current: jax.Array,
            soma_potential: jax.Array | None,
            bap_conductance: jax.Array,
        ) -> jax.Array:
        """
            Membrane integration. The only part of the step every model has to provide.

            Parameters
            ----------
            potential : jax.Array
                Dendritic potential at the start of the step, relative to rest.
            current : jax.Array
                Current delivered to the dendrite, in pA.
            soma_potential : jax.Array or None
                Somatic potential in the dendritic frame, or None when the dendrite is isolated.
            bap_conductance : jax.Array
                Back-propagating conductance on this step, in nS, towards ``bap_reversal``.
        """
        pass

    def _effective_current(self, current: jax.Array) -> jax.Array:
        """
            The current the membrane actually integrates.
        """
        return current

    def _update_state(self, potential: jax.Array, new_potential: jax.Array) -> None:
        """
            Called once the membrane has been integrated. State that follows the potential, such
            as gating variables, is updated here.
        """
        pass

    def _effective_threshold(self) -> jax.Array:
        """
            The value the membrane potential is tested against.
        """
        return self.threshold.value

    def _plateau_computation(self, potential: jax.Array, threshold: jax.Array) -> jax.Array:
        """
            How far the membrane is into a plateau.

            A smooth reading of the threshold crossing, so that the plateau can drive a trace or
            gate a plasticity rule without a hard switch. It equals 0.5 at the threshold, so
            thresholding the result at 0.5 recovers the hard flag exactly.
        """
        return jax.nn.sigmoid((potential - threshold) / self.plateau_slope.value)

    def _soma_current(self, potential: jax.Array, soma_potential: jax.Array | None) -> jax.Array:
        """
            Axial current delivered to the soma, in pA.
        """
        if soma_potential is None:
            return jnp.zeros_like(potential)
        return self.coupling.value * (potential - soma_potential)

    def _bap_conductance(self, spikes: jax.Array | None) -> jax.Array:
        """
            Back-propagating conductance on this step, in nS.

            Spikes feed a rise and a decay trace whose difference is the double exponential
            kernel. The kernel is zero on the step a spike arrives and acts from the next one.
        """
        if spikes is None:
            return jnp.zeros((), dtype=self._dtype)
        arrival = spikes.astype(self._dtype)
        rise = self._bap_rise.value - self.bap_rise_rate.value * self._bap_rise.value + arrival
        fall = self._bap_fall.value - self.bap_fall_rate.value * self._bap_fall.value + arrival
        self._bap_rise.value = rise
        self._bap_fall.value = fall
        return self.bap_conductance.value * (fall - rise)

    def __call__(
            self,
            in_current: CurrentArray,
            soma_potential: PotentialArray | None = None,
            spikes: SpikeArray | None = None,
            injected_current: CurrentArray | None = None,
        ) -> DendriteOutput:
        """
            Advances the dendrite one step.

            Parameters
            ----------
            in_current : CurrentArray
                Current delivered to the dendrite, in pA.
            soma_potential : PotentialArray, optional
                Potential of the coupled soma, relative to the soma's rest.
            spikes : SpikeArray, optional
                Somatic spikes, driving the back-propagating action potential.
            injected_current : CurrentArray, optional
                Additional current injected into the dendrite, in pA.

            Returns
            -------
            DendriteOutput
                Dictionary with ``out_current`` and ``plateau``.
        """
        potential = self._potential.value
        # The soma potential is moved into the dendritic frame.
        soma = None if soma_potential is None else soma_potential.value.astype(self._dtype) + self.soma_offset.value
        # The spike bit alone: a back-propagating spike is unsigned.
        bap = self._bap_conductance(None if spikes is None else spikes.spikes)
        # Membrane.
        current = in_current.value.astype(self._dtype)
        if injected_current is not None:
            current = current + injected_current.value.astype(self._dtype)
        current = self._effective_current(current)
        new_potential = self._integrate(potential, current, soma, bap)
        self._potential.value = new_potential
        self._update_state(potential, new_potential)
        # Outputs.
        plateau = self._plateau_computation(new_potential, self._effective_threshold())
        out_current = self._soma_current(new_potential, soma)
        return {
            'out_current': CurrentArray(value=out_current),
            'plateau': FloatArray(value=plateau),
        }

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
