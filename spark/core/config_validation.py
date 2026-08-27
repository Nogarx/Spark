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
        Context manager suspending the field validators.

        For building a configuration out of values that are not valid on their own yet, such as a
        half-finished model in the graph editor.

        Examples
        --------
        >>> with NoValidation():
        ...     config = LeakySomaConfig(potential_tau=None)

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
        Base class for the validators of a configuration field.

        Parameters
        ----------
        field : dataclasses.Field
            Field being guarded.
        valid_types : tuple of type, optional
            Types the field accepts. Read from the field metadata when omitted.
    """

    def __init__(self, field: dc.Field, valid_types: tuple[tp.Any, ...] | None = None) -> None:
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
        Checks the value against the types declared by the field.

        The types come from the ``valid_types`` metadata entry, which the metaclass fills in from
        the annotation.
    """

    def validate(self, value: tp.Any) -> None:
        valid_types = self._types()
        # NOTE: An unreadable annotation yields no types, and validating against none of them would
        # refuse every value. The field is left unchecked instead.
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
            The types this field accepts.

            Returns
            -------
            tuple of type or None
                None when the annotation could not be read, in which case the field is left
                unchecked.
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
        Checks that every entry of the value is greater than zero.
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
        Checks that every entry of the value is 0 or 1.
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
        Checks that every entry of the value lies in ``[0, 1]``.
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