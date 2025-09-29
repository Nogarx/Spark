#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax
import numpy as np
import jax.numpy as jnp
import dataclasses as dc
import typing as tp
from spark.core.registry import REGISTRY
from spark.core.registry import register_cfg_validator

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ConfigurationValidator:

    def __init__(self, field: dc.Field):
        self.field = field

    @abc.abstractmethod
    def validate(self,) -> None:
        pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: TypeValidator is very convoluted and brittle due to valid_types string format.
# It is likely that there is a better way to implement this logic method.
# TODO: There is some room to relax type checking in some contexts. 
# Some castings will drastically improve user experience. Currently allowing int to float. 
# TODO: Add support for sensible numpy and jax.numpy validation.
@register_cfg_validator
class TypeValidator(ConfigurationValidator):

    def validate(self, value: tp.Any) -> None:
        # Get and parse valid types.
        valid_types = self.field.metadata.get('valid_types')
        valid_types = self._str_to_types(valid_types)
        # Compare against valid types.
        if len(valid_types) > 0 and not isinstance(value, valid_types):
            if len(valid_types) == 1:
                types_str = valid_types[0].__name__
            else:
                types_str = ', '.join([f'\"{t.__name__}\"' for t in valid_types[:-1]]) + f' or \"{valid_types[-1].__name__}\"'
            raise TypeError(f'Attribute "{self.field.name}" expects types {types_str}, but got type \"{type(value).__name__}\".')

    @staticmethod    
    def _str_to_types(str_types: str):
        str_list = str_types.replace(' ', '').split('|')
        types_list = []
        for st in str_list:
            t = None
            # Check built-in classes
            if not t:
                t = globals()['__builtins__'].get(st)
                if t:
                    types_list.append(t)
                    # Manual int to float promotion
                    if t == float:
                        types_list.append(int)
            # Check if DTypeLike
            if not t:
                if st == 'DTypeLike':
                    t = np.dtype
                    types_list.append(str)
                    types_list.append(np.dtype)
                    types_list.append(jnp.dtype)
                    types_list.append(jax._src.typing.SupportsDType)
            # Check if it is a spark class
            if not t:
                t = REGISTRY.MODULES.get(st)
                if t:
                    types_list.append(t)
            if not t:
                t = REGISTRY.PAYLOADS.get(st)
                if t:
                    types_list.append(t)
            if not t:
                t = REGISTRY.INITIALIZERS.get(st)
                if t:
                    types_list.append(t)
        return tuple(types_list)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_cfg_validator
class PositiveValidator(ConfigurationValidator):

    def validate(self, value: tp.Any) -> None:
        if isinstance(value, (int, float)):
            is_positive = value > 0
        elif isinstance(value, jnp.ndarray):
            is_positive = jnp.all(value > 0)
        elif isinstance(value, np.ndarray):
            is_positive = np.all(value > 0)
        else:
            raise TypeError(f'value is not a supported numeric object.')
        if not is_positive:
            raise ValueError(f'Attribute "{self.field.name}" must be positive, but got {value}.')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_cfg_validator
class BinaryValidator(ConfigurationValidator):

    def validate(self, value: tp.Any) -> None:
        if isinstance(value, (int, float)):
            is_zero_one = value == 0 or value == 1
        elif isinstance(value, jnp.ndarray):
            is_zero_one = jnp.all(jnp.logical_or(value == 0, value == 1))
        elif isinstance(value, np.ndarray):
            is_zero_one = np.all(np.logical_or(value == 0, value == 1))
        elif isinstance(value, bool):
            is_zero_one = True
        else:
            raise TypeError(f'value is not a supported binary numeric object.')
        if not is_zero_one:
            raise ValueError(f'Attribute "{self.field.name}" values must be binary (0/1 values), but got {value}.')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_cfg_validator
class ZeroOneValidator(ConfigurationValidator):

    def validate(self, value: tp.Any) -> None:
        if isinstance(value, (int, float)):
            is_zero_one = value >= 0 and value <= 1
        elif isinstance(value, jnp.ndarray):
            is_zero_one = jnp.all(jnp.logical_and(value >= 0, value <= 1))
        elif isinstance(value, np.ndarray):
            is_zero_one = np.all(np.logical_and(value >= 0, value <= 1))
        else:
            raise TypeError(f'value is not a supported numeric object.')
        if not is_zero_one:
            raise ValueError(f'Attribute "{self.field.name}" values must be in the range [0,1], but got {value}.')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################