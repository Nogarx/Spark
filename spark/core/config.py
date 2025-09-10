#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import abc
import jax.numpy as jnp
import dataclasses
import typing as tp
from jax.typing import DTypeLike
from typing import Any
from collections import OrderedDict
from spark.core.config_validation import TypeValidator, PositiveValidator

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# TODO: Add some logic to the metadata for auto-inference of certain parameters when using GUI.
# For example, some Shapes can be infered from the next/prev component in the graph.
METADATA_TEMPLATE = {
    'units': None, 
    'valid_types': Any, 
    'validators': [], 
    'description': '',
    }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ConfigurationError(BaseException):
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkMetaConfig(abc.ABCMeta):
    """
        Metaclass that promotes class attributes to dataclass fields
    """
    
    def __new__(cls, name: str, bases: tuple[type, ...], dct: dict[str, Any]) -> 'SparkMetaConfig':

        # NOTE: Every non field is promoted to field to simplify the logic of configuration objects and add metadata.
        
        # Iterate over annotations.
        annotations: dict[str, Any] = dct.get('__annotations__', {})
        for attr_name, attr_type in annotations.items():
            # Parse valid types.
            if isinstance(attr_type, str):
                valid_types = {'valid_types': attr_type}
            elif tp.get_origin(attr_type):
                attr_types = list(tp.get_args(attr_type))
                try:
                    attr_types.remove(type(None))
                except:
                    pass
                attr_types = '|'.join([ft.__name__ for ft in attr_types])
                valid_types = {'valid_types': attr_types}
            else:
                valid_types = {'valid_types': attr_type.__name__}

            # Get default value
            attr_value = dct.get(attr_name, dataclasses.MISSING)

            # If is field, add template info
            if isinstance(attr_value, dataclasses.Field):
                # Merge metadata with template.
                validators = {'validators': list(set(METADATA_TEMPLATE['validators'] + 
                                                     attr_value.metadata['validators']) if 'validators' in attr_value.metadata else [])}
                attr_value.metadata = {**METADATA_TEMPLATE, **(attr_value.metadata or {}), **valid_types, **validators}
                field = attr_value

            # If is not field, promote to field
            else:
                # Create a new Field object.
                field = dataclasses.field(
                    default=attr_value,
                    metadata={**METADATA_TEMPLATE, **valid_types}
                )

            # If is config, ...
            if getattr(getattr(attr_value, 'default_factory', False), '__is_spark_config__', False):
                print(field.default_factory.__annotations__)

            # Update dct
            dct[attr_name] = field

        # Create the dataclass
        new_class = super().__new__(cls, name, bases, dct)
        new_dataclass = dataclasses.dataclass(new_class, kw_only=True)
        return new_dataclass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class BaseSparkConfig(abc.ABC, metaclass=SparkMetaConfig):
    """
        Base class for module configuration.
    """
    __schema_version__ = '1.0'
    __config_delimiter__ = '__'
    __shared_config_delimiter__ = 'S__'
    __is_spark_config__ = True

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
            partial = cls._fold_partial(instance, partial)
            # Get current fields and pass attributes.
            instance._set_partial_attributes(instance, partial)
        # Validate
        instance.validate()
        return instance

    def merge(self, partial: dict[str, Any]) -> None:
        """
            Update config with partial overrides.
        """
        # Fold partial to pass child config attributes.
        partial = self._fold_partial(self, partial)
        # Get current fields and pass attributes.
        self._set_partial_attributes(self, partial)
        # Validate
        self.validate()

    @staticmethod
    def _fold_partial(obj: 'BaseSparkConfig', partial: dict[str, Any]) -> dict[str, Any]:
        fold_partial = {}
        for key, value in partial.items():
            if obj.__config_delimiter__ in key:
                child_key, nested_key = key.split(obj.__config_delimiter__, maxsplit=1)
                if child_key not in fold_partial:
                    fold_partial[child_key] = {}
                fold_partial[child_key][nested_key] = value
            else:
                fold_partial[key] = value
        return fold_partial
    
    @staticmethod
    def _set_partial_attributes(obj: 'BaseSparkConfig', partial: dict[str, Any]) -> None:
        valid_fields = {field.name: field for field in dataclasses.fields(obj) }
        for key, value in partial.items():
            if key in valid_fields:
                field_type = valid_fields[key].type
                # Field is another SparkConfig.
                if getattr(field_type, '__is_spark_config__', None):
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

    def validate(self,) -> None:
        for field in dataclasses.fields(self):
            for validator in field.metadata.get('validators', []):
                validator_instance = validator(field)
                validator_instance.validate(getattr(self, field.name))

    def get_metadata(self) -> dict[str, Any]:
        metadata = {}
        for field in dataclasses.fields(self):
            metadata[field.name] = dict(field.metadata)
        return metadata
    
    def __post_init__(self,):
        self.validate()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkConfig(BaseSparkConfig):
    """
        Default class for module configuration.
    """
    seed: int = dataclasses.field(
        default_factory=lambda: int.from_bytes(os.urandom(4), 'little'), 
        metadata={
            'validators': [
                TypeValidator,
            ], 
            'description': 'Seed for internal random processes.',
        })
    dtype: DTypeLike = dataclasses.field(
        default=jnp.float16, 
        metadata={
            'validators': [
                TypeValidator,
            ], 
            'description': 'Dtype used for JAX dtype promotions.',
        })
    dt: float = dataclasses.field(
        default=1.0, 
        metadata={
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Deltatime integration constant.',
        })
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################