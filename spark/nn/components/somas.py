#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax.numpy as jnp
from math import prod
from typing import TypedDict, Dict, List
from dataclasses import dataclass
from spark.core.tracers import Tracer
from spark.core.shape import bShape, Shape, normalize_shape
from spark.core.payloads import SpikeArray, CurrentArray, PotentialArray
from spark.core.variable_containers import Variable, Constant, ConfigDict
from spark.core.registry import register_module
from spark.nn.components.base import Component

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@dataclass
class ALIFConfigurations:
    DEFAULT = {
        'potential_rest':-60.0,     # [mV]
        'potential_reset':-50.0,    # [mV]  
        'potential_tau':20.0,       # [ms]
        'conductance':10.0,         # [nS] <-> [1/GΩ]
        'threshold':-40.0,          # [mV]
        'threshold_tau':100.0,      # [ms]
        'threshold_delta':100.0,    # [mV]
        'cooldown':2.0,             # [ms]
    }
    LIF = {
        'potential_rest':-60.0,     # [mV]
        'potential_reset':-50.0,    # [mV]  
        'potential_tau':20.0,       # [ms]
        'conductance':10.0,         # [nS] <-> [1/GΩ]
        'threshold':-40.0,          # [mV]
        'threshold_tau':1.0,        # [ms]
        'threshold_delta':0.0,      # [mV]
        'cooldown':2.0,             # [ms]
    }
ALIF_cfgs = ALIFConfigurations()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Generic Soma output contract.
class SomaOutput(TypedDict):
    spikes: SpikeArray
    potentials: PotentialArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Soma(Component):
    """
        Abstract soma model.
    """

    def __init__(self, 
                 shape: bShape,
                 params: Dict = {},
                 **kwargs):
        # Initialize super.
        super().__init__(**kwargs)
        # Initialize shapes
        self._shape = normalize_shape(shape)
        self.units = prod(self._shape)
        # Initialize variables
        self._params = params
        self._potential = Variable(jnp.zeros(self._shape, dtype=self._dtype), dtype=self._dtype)

    @property
    def potential(self,) -> PotentialArray:
        """
            Returns the current soma potential.
        """
        return PotentialArray(self._potential.value)
    
    @property
    def input_shapes(self,) -> List[Shape]:
        return [self._shape]

    @property
    def output_shapes(self,) -> List[Shape]:
        return [self._shape]

    @abc.abstractmethod
    def reset(self):
        """
            Resets neuron states to their initial values.
        """
        pass

    @abc.abstractmethod
    def _update_states(self, current: CurrentArray) -> None:
        """
            Update neuron's states variables.
        """
        pass

    @abc.abstractmethod
    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        pass

    def __call__(self, current: CurrentArray) -> SomaOutput:
        """
            Update neuron's states and compute spikes.
        """
        self._update_states(current)
        return {
            'spikes': self._compute_spikes(), 
            'potentials': self.potential,
        }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_module
class ALIFSoma(Soma):
    """
        LIF soma model.
    """

    def __init__(self, 
                 shape: bShape,
                 params: Dict = {},
                 **kwargs):
        # Initialize super.
        super().__init__(shape, params=params, **kwargs)
        # Set missing parameters to default values.
        for k in ALIF_cfgs.DEFAULT:
            self._params.setdefault(k,  ALIF_cfgs.DEFAULT[k])
        self._params = ConfigDict(config=self._params)
        # Initialize parameters.
        # Membrane. Substract potential_rest to potential related terms to rebase potential at zero.
        self._potential_reset = Constant(self._params['potential_reset'] - self._params['potential_rest'], dtype=self._dtype)
        self._potential_scale = Constant(self._dt / self._params['potential_tau'], dtype=self._dtype)
        # Conductance.
        self._conductance = Constant(1/self._params['conductance'], dtype=self._dtype)
        # Threshold.
        self._threshold = Tracer(self._shape,
                                 tau=self._params['threshold_tau'], 
                                 base=(self._params['threshold'] - self._params['potential_rest']), 
                                 scale=self._params['threshold_delta'], 
                                 dt=self._dt, dtype=self._dtype)
        # Refractory period.
        self.cooldown = Constant(self._params['cooldown'], dtype=self._dtype)
        self.refractory = Variable(jnp.array(self.cooldown), dtype=self._dtype)

        

    def reset(self) -> None:
        """
            Resets component state.
        """
        self._potential.value = jnp.zeros_like(self._shape, dtype=self._dtype)
        self.refractory.value = jnp.array(self.cooldown, dtype=self._dtype)
        self._threshold.reset()

    def _update_states(self, I: CurrentArray) -> None:
        """
            Update neuron's soma states variables.
        """
        is_ready = jnp.greater(self.refractory.value, self.cooldown).astype(self._dtype)
        self._potential.value += self._potential_scale * (-self._potential.value + is_ready * self._conductance * I.value)

    def _compute_spikes(self,) -> SpikeArray:
        """
            Compute neuron's spikes.
        """
        # Compute spikes.
        spikes = jnp.logical_and(jnp.greater(self._potential.value, self._threshold.value), 
                                 jnp.greater(self.refractory.value, self.cooldown)).astype(self._dtype)
        # Reset neurons.
        self._potential.value = spikes * self._potential_reset + (1 - spikes) * self._potential.value
        # Update thresholds
        self._threshold(spikes)
        # Set neuron refractory period.
        self.refractory.value = (1 - spikes) * (self.refractory.value + self._dt)
        return SpikeArray(spikes)
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################