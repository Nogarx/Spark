#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax
import numpy as np
import jax.numpy as jnp
import jax.typing
import dataclasses as dc
import typing as tp
from spark.core.registry import REGISTRY, register_cfg_validator, RegistryNamespace
from spark.core.signature_parser import is_instance, normalize_typehint

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class NoValidation:
    """
        Context manager suspending the validators, for the cases where a configuration is knowingly built out
        of values that do not stand on their own yet.
    """

    def __enter__(self) -> 'NoValidation':
        from spark.core.config import validation_enabled, set_validation
        self._previous = validation_enabled()
        set_validation(False)
        return self

    def __exit__(self, *exception) -> None:
        from spark.core.config import set_validation
        set_validation(self._previous)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ConfigurationValidator:
    """
        Base class for validators for the fields in a SparkConfig.
    """

    def __init__(self, field: dc.Field, valid_types: tuple[tp.Any, ...] | None = None) -> None:
        """
            Args:
                field: dc.Field, field being guarded.
                valid_types: tuple[type, ...], types the field accepts
        """
        self.field = field
        self.valid_types = valid_types

    @abc.abstractmethod
    def validate(self, value: tp.Any) -> None:
        pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: There is some room to relax type checking in some contexts.
# Some castings will drastically improve user experience. Currently allowing int to float.
@register_cfg_validator
class TypeValidator(ConfigurationValidator):
    """
        Validates the type of the field against a set of valid_types defined in the metadata.
    """

    def validate(self, value: tp.Any) -> None:
        valid_types = self._types()
        # NOTE: An annotation this validator cannot read is a question it cannot answer, and answering it
        # anyway would refuse every value of the field. Silence is the only correct verdict there.
        if not valid_types:
            return
        if is_instance(value, valid_types):
            return
        names = [getattr(t, '__name__', None) or str(t) for t in valid_types]
        types_str = names[0] if len(names) == 1 else ', '.join(f'"{n}"' for n in names[:-1]) + f' or "{names[-1]}"'
        raise TypeError(
            f'Attribute "{self.field.name}" expects types {types_str}, but got type \"{type(value).__name__}\".'
        )

    def _types(self) -> tuple[tp.Any, ...]:
        """
            The types this field accepts or None if they cannot be read.
        """
        valid_types = self.valid_types if self.valid_types is not None else self.field.metadata.get('valid_types')
        valid_types = tuple(valid_types or ())
        if not valid_types or any(isinstance(t, str) for t in valid_types):
            return ()
        if float in valid_types and int not in valid_types:
            valid_types = valid_types + (int,)
        if jax.Array in valid_types and np.ndarray not in valid_types:
            valid_types = valid_types + (np.ndarray,)
        return valid_types

@register_cfg_validator
class PositiveValidator(ConfigurationValidator):
    """
        Validates that the value(s) of the attribute are greater than zero.
    """

    def validate(self, value: tp.Any) -> None:
        if isinstance(value, (int, float)):
            is_positive = value > 0
        elif isinstance(value, (jnp.ndarray, np.ndarray)):
            is_positive = bool(np.all(np.asarray(value) > 0))
        elif isinstance(value, (tuple, list, set)):
            is_positive = bool(np.all(np.asarray(list(value)) > 0))
        else:
            raise TypeError(f'{value} is not a supported numeric object.')
        if not is_positive:
            raise ValueError(f'Attribute "{self.field.name}" must be positive, but got {value}.')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_cfg_validator
class BinaryValidator(ConfigurationValidator):
    """
        Validates that the value(s) of the attribute are in the set {0,1}.
    """

    def validate(self, value: tp.Any) -> None:
        if isinstance(value, bool):
            is_zero_one = True
        elif isinstance(value, (int, float)):
            is_zero_one = value == 0 or value == 1
        elif isinstance(value, (jnp.ndarray, np.ndarray)):
            array = np.asarray(value)
            is_zero_one = bool(np.all(np.logical_or(array == 0, array == 1)))
        elif isinstance(value, (tuple, list, set)):
            array = np.asarray(list(value))
            is_zero_one = bool(np.all(np.logical_or(array == 0, array == 1)))
        else:
            raise TypeError(f'{value} is not a supported binary numeric object.')
        if not is_zero_one:
            raise ValueError(f'Attribute "{self.field.name}" values must be binary (0/1 values), but got {value}.')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_cfg_validator
class ZeroOneValidator(ConfigurationValidator):
    """
        Validates that the value(s) of the attribute are in the range [0,1].
    """

    def validate(self, value: tp.Any) -> None:
        if isinstance(value, (int, float)):
            is_zero_one = value >= 0 and value <= 1
        elif isinstance(value, (jnp.ndarray, np.ndarray)):
            array = np.asarray(value)
            is_zero_one = bool(np.all(np.logical_and(array >= 0, array <= 1)))
        elif isinstance(value, (tuple, list, set)):
            array = np.asarray(list(value))
            is_zero_one = bool(np.all(np.logical_and(array >= 0, array <= 1)))
        else:
            raise TypeError(f'value is not a supported numeric object.')
        if not is_zero_one:
            raise ValueError(f'Attribute "{self.field.name}" values must be in the range [0,1], but got {value}.')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################