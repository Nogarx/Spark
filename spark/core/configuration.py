#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import abc
import numpy as np
import jax.numpy as jnp
import dataclasses
from jax.typing import DTypeLike
from typing import Any, Callable
from collections import OrderedDict

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ConfigurationValidator:

    def __init__(self, field: dataclasses.Field):
        self.field = field

    @abc.abstractmethod
    def validate(self,):
        pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TypeValidator(ConfigurationValidator):

    def validate(self, value: Any) -> None:
        valid_types = self.field.metadata.get('valid_types')
        if valid_types and not isinstance(value, valid_types):
            raise TypeError(f'Attribute "{self.field.name}" expects types {valid_types}, but got type {type(value).__name__}.')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class PositiveValidator(ConfigurationValidator):

    def validate(self, value: Any) -> None:
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

class BinaryValidator(ConfigurationValidator):

    def validate(self, value: Any) -> None:
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

class ZeroOneValidator(ConfigurationValidator):

    def validate(self, value: Any) -> None:
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

# TODO: Add some logic to the metadata for auto-inference of certain parameters when using GUI.
# For example, some Shapes can be infered from the next/prev component in the graph.

METADATA_TEMPLATE = {
    'units': None, 
    'valid_types': Any, 
    'validators': [TypeValidator], 
    'description': '',
    }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Meta module to resolve metaclass conflicts
class SparkMetaConfig(type):
    """
        Metaclass that promotes class attributes to dataclass fields
    """
    
    def __new__(cls, name: str, bases: tuple[type, ...], dct: dict[str, Any]) -> 'SparkMetaConfig':

        # NOTE: We need to manually intercept the init annotations since non-default arguments may follow 
        # default arguments when extending configuration classes. This is achieved by "flattening" the 
        # inheritance hierarchy, leaving the leaf dataclass as the only element in the MRO, not ideal
        # but no a real problem.

        # NOTE: Every non field is promoted to field to simplify the logic of configuration objects.

        # Flatten attributes
        fields = OrderedDict()
        for base in bases:
            # Gather fields
            for field in dataclasses.fields(base):
                field_info = base.__dataclass_fields__[field.name]
                fields[field.name] = (field.type, field_info)
            # Gather methods and properties
            for attr_name, attr_value in base.__dict__.items():
                if not attr_name.startswith('__') and (
                        callable(attr_value) 
                        or isinstance(attr_value, property)
                        or isinstance(attr_value, classmethod)
                        or isinstance(attr_value, staticmethod)):
                    dct[attr_name] = attr_value

        # Gather fields from the current class. Prioritize child's attributes over parent's.
        current_annotations = dct.get('__annotations__', {})
        for field_name, field_type in current_annotations.items():
            field_value = dct.get(field_name, dataclasses.MISSING)
            fields[field_name] = (field_type, field_value)

        # Process and standardize all fields with metadata
        processed_fields = OrderedDict()
        for field_name, (field_type, field_info) in fields.items():
            valid_types = {'valid_types': field_type}
            if isinstance(field_info, dataclasses.Field):
                # Merge metadata with template.
                validators = {'validators': list(set(METADATA_TEMPLATE['validators'] + 
                                                     field_info.metadata['validators']) if 'validators' in field_info.metadata else [])}
                field_info.metadata = {**METADATA_TEMPLATE, **(field_info.metadata or {}), **valid_types, **validators}
                field = field_info
            else:
                # Create a new Field object.
                field = dataclasses.field(
                    default=field_info if field_info is not dataclasses.MISSING else dataclasses.MISSING,
                    metadata={**METADATA_TEMPLATE, **valid_types}
                )
            processed_fields[field_name] = (field_type, field)

        # Reorder the standardized fields
        non_default_fields = OrderedDict()
        default_fields = OrderedDict()
        for field_name, (field_type, field_obj) in processed_fields.items():
            if field_obj.default is dataclasses.MISSING and \
               field_obj.default_factory is dataclasses.MISSING:
                non_default_fields[field_name] = (field_type, field_obj)
            else:
                default_fields[field_name] = (field_type, field_obj)
        
        # Rebuild attributes for the new class
        annotations = OrderedDict()
        for field_name, (field_type, field_obj) in non_default_fields.items():
            annotations[field_name] = field_type
            dct[field_name] = field_obj
        for field_name, (field_type, field_obj) in default_fields.items():
            annotations[field_name] = field_type
            dct[field_name] = field_obj
        dct['__annotations__'] = annotations
        
        # Create the dataclass
        new_class = super().__new__(cls, name, tuple([]), dct)
        new_class.__is_spark_config__ = True
        return dataclasses.dataclass(new_class)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class BaseSparkConfig(abc.ABC, metaclass=SparkMetaConfig):
    """
        Base class for module configuration.
    """
    __SCHEMA_VERSION__ = '1.0'
    __CONFIG_DELIMITER__ = '__'
    __SHARED_CONFIG_DELIMITER__ = 'S__'

    @classmethod
    def create(cls: type['BaseSparkConfig'], partial: dict[str, Any] = None) -> 'BaseSparkConfig':
        """
            Create config with partial overrides.
        """
        # Get default instance
        instance = cls()
        # Apply partial updates
        if partial:

            # Fold partial to pass child config attributes.
            partial = cls._fold_partial(partial)
            # Get current fields and pass attributes.
            cls._set_partial_attributes(cls, partial)

        return instance

    def merge(self, partial: dict[str, Any]) -> None:
        """
            Update config with partial overrides.
        """
        # Fold partial to pass child config attributes.
        partial = self._fold_partial(partial)
        # Get current fields and pass attributes.
        self._set_partial_attributes(self, partial)

    @staticmethod
    def _fold_partial(obj: BaseSparkConfig, partial: dict[str, Any]) -> dict[str, Any]:
        fold_partial = {}
        for key, value in partial.items():
            if obj.__CONFIG_DELIMITER__ in key:
                child_key, nested_key = key.split('__', maxsplit=1)
                if child_key not in fold_partial:
                    fold_partial[child_key] = {}
                fold_partial[child_key][nested_key] = value
            else:
                fold_partial[key] = value
        return 
    
    @staticmethod
    def _set_partial_attributes(obj: BaseSparkConfig, partial: dict[str, Any]) -> None:
        valid_fields = {field.name: field for field in dataclasses.fields(obj) }
        for key, value in partial.items():
            if key in valid_fields:
                field_type = valid_fields[key].type
                # Field is another SparkConfig.
                if issubclass(field_type, BaseSparkConfig):
                    child_config = field_type.create(value)
                    setattr(obj, key, child_config)
                # Field is a plain value.
                else:
                    setattr(obj, key, value)
            else:
                raise ValueError(f'Invalid config key: {key}')


    def diff(self, other: 'SparkConfig') -> dict[str, Any]:
        """
            Return differences from another config.
        """
        return {
            field.name: getattr(self, field.name) 
            for field in dataclasses.fields(self) 
            if getattr(self, field.name) != getattr(other, field.name)
        }

    @classmethod
    def from_dict(cls: type['SparkConfig'], data: dict[str, Any]) -> 'SparkConfig':
        """
            Create config instance from dictionary
        """
        return cls(**data)
    
    def to_dict(self) -> dict[str, dict[str, Any]]:
        """
            Serialize config to dictionary
        """
        return {
            field.name: {
                'value': getattr(self, field.name),
                'metadata': field.metadata} 
            for field in dataclasses.fields(self) 
        }

    def validate(self) -> None:
        for field in dataclasses.fields(self):
            for validator in field.metadata.get('validators', []):
                validator_instance = validator(field)
                validator_instance.validate(getattr(self, field.name))

    def get_metadata(self) -> dict[str, Any]:
        metadata = {}
        for field in dataclasses.fields(self):
            metadata[field.name] = dict(field.metadata)
        return metadata

    def __post_init__(self):
        self.validate()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkConfig(BaseSparkConfig):
    """
        Default class for module configuration.
    """
    seed: int = dataclasses.field(
        default_factory=lambda: int.from_bytes(os.urandom(4), 'little'), 
        metadata={
            'description': 'Seed for internal random processes.',
        })
    dtype: DTypeLike = dataclasses.field(
        default=jnp.float16, 
        metadata={
            'description': 'Dtype used for JAX dtype promotions.',
        })
    dt: float = dataclasses.field(
        default=1.0, 
        metadata={
            'units': 'ms',
            'validators': [
                PositiveValidator,
            ],
            'description': 'Deltatime integration constant.',
        })
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################