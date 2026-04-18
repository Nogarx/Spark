#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import typing as tp
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from spark.core.specs import PortSpecs
from spark.core.variables import Constant
from spark.core.utils import get_einsum_dot_exp_string
from spark.core.payloads import FloatArray, IntegerMask
from spark.nn.components.base import Component, ComponentConfig
from spark.nn.initializers import MaskedInitializer

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

PlasticityParamLike = float  | tuple[float, float, float, float] | jax.Array #| MaskedInitializer

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class PlasticityOutput(tp.TypedDict):
    """
       Generic plasticity rule model output spec.
    """
    kernel: FloatArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class PlasticityConfig(ComponentConfig):
    """
        Abstract plasticity rule configuration class.
    """
    pass
ConfigT = tp.TypeVar("ConfigT", bound=PlasticityConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Plasticity(Component, tp.Generic[ConfigT]):
    """
        Abstract plasticity rule model.
    """

    def __init__(self, config: ConfigT | None = None, **kwargs) -> None:
        # Initialize super.
        super().__init__(config = config, **kwargs)

    @property
    def synaptic_mask(self,) -> IntegerMask:
        if getattr(self, '_synaptic_mask', None) is None:
            return IntegerMask(jnp.array(0))
        else:
            return IntegerMask(self._synaptic_mask.value)
        
    @property
    def synaptic_mask_map(self,) -> dict[int, str]:
        return {
            0: 'EE',
            1: 'EI',
            2: 'IE',
            3: 'II',
        }

    def _initialize_synaptic_mask(self, input_specs: dict[str, PortSpecs]) -> None:
        # Generate mask once
        if getattr(self, '_synaptic_mask', None) is None:
            post_inhibition_mask = input_specs['post_spikes'].inhibition_mask
            if input_specs['pre_spikes'].async_spikes: 
                _slice = post_inhibition_mask.ndim * (0,) + (input_specs['pre_spikes'].inhibition_mask.ndim - post_inhibition_mask.ndim) * (slice(None),)
                pre_inhibition_mask = input_specs['pre_spikes'].inhibition_mask[_slice]
            else:
                pre_inhibition_mask = input_specs['pre_spikes'].inhibition_mask 
            einsum_str = get_einsum_dot_exp_string(
                post_inhibition_mask.shape, 
                pre_inhibition_mask.shape, 
                side='none'
            )
            synaptic_mask = (
                + 0*jnp.einsum(einsum_str, 1-post_inhibition_mask, 1-pre_inhibition_mask)    # EE
                + 1*jnp.einsum(einsum_str, 1-post_inhibition_mask, pre_inhibition_mask)      # EI
                + 2*jnp.einsum(einsum_str, post_inhibition_mask, 1-pre_inhibition_mask)      # IE
                + 3*jnp.einsum(einsum_str, post_inhibition_mask, pre_inhibition_mask)        # II
            )
            self._synaptic_mask = Constant(synaptic_mask, dtype=jnp.uint8)

    def _initialize_variable(self, variable: PlasticityParamLike, shape: tuple, dtype: jnp.dtype) -> jax.Array:
        """
            Parses a config parameter into a JAX array matching the synapse configuration
        """
        # Resolve variable raw value
        if callable(variable):
            _var = variable(key=self.get_rng_keys(1), shape=shape, dtype=dtype)
        else:
            _var = variable
        # Construct jax.Array
        # Case 1) variable is a tuple of connection type specific values  
        if isinstance(_var, (tuple, list)) and len(_var) == 4:
            # Handle Tensor (EE, EI, IE, II) resolution
            _val = jnp.zeros_like(self._synaptic_mask.value, dtype=dtype)
            for i in range(4):
                _val = jnp.where(self._synaptic_mask.value == i, _var[i], _val)
            return _val
        # Case 2) variable is a scalar or is already an array
        elif isinstance(_var, ArrayLike):
            return jnp.array(_var, dtype=dtype)
        # Case 3) variable is who knows what 
        else:
            raise TypeError(
                f'Cannot resolve "variable" to and ArrayLike or an iterable of size four. Got: {_var}'
            )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################