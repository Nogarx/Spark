#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import abc
import jax
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

            # Update dct
            dct[attr_name] = field

        # Create the dataclass
        new_class = super().__new__(cls, name, bases, dct)
        new_dataclass = dataclasses.dataclass(new_class, kw_only=True, init=False)
        return new_dataclass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_static
@dataclasses.dataclass(init=False, frozen=True, kw_only=True)
class FrozenSparkConfig(abc.ABC, metaclass=SparkMetaConfig):
    """
        Frozen class for module configuration compatible with JIT.
    """

    def __init__(self, spark_config: SparkConfig):
        if not issubclass(type(spark_config), BaseSparkConfig):
            raise TypeError(f'"spark_config" must be of a subclass of "BaseSparkConfig", got "{type(spark_config).__name__}"')
        
        # Get type hints
        type_hints = tp.get_type_hints(spark_config.__class__)
        for key in type_hints.keys():
            if tp.get_origin(type_hints[key]): 
                hints = list(tp.get_args(type_hints[key]))
                type_hints[key] = (h for h in hints if h is not type(None))

        for field in dataclasses.fields(spark_config):
            if isinstance(type_hints[field.name], type) and issubclass(type_hints[field.name], BaseSparkConfig):
                # Freeze configs recursively
                object.__setattr__(self, f'{field.name}', FrozenSparkConfig(getattr(spark_config, f'{field.name}')))
            else:
                # Plain attribute
                object.__setattr__(self, f'{field.name}', getattr(spark_config, f'{field.name}'))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class BaseSparkConfig(abc.ABC, metaclass=SparkMetaConfig):
    """
        Base class for module configuration.
    """
    __schema_version__ = '1.0'
    __config_delimiter__ = '__'
    __shared_config_delimiter__ = '_s_'

    def __init__(self, **kwargs):

        # Get type hints
        type_hints = tp.get_type_hints(self.__class__)
        for key in type_hints.keys():
            if tp.get_origin(type_hints[key]): 
                hints = list(tp.get_args(type_hints[key]))
                type_hints[key] = (h for h in hints if h is not type(None))

        # Fold kwargs
        kwargs_fold, shared_partial = self._fold_partial(self, kwargs)

        # Set attributes programatically
        for field in dataclasses.fields(self.__class__):

            # Parse field name, remove __shared_config_delimiter__ if present
            if self.__shared_config_delimiter__ == field.name[:len(self.__shared_config_delimiter__)]:
                field_name = key[len(self.__shared_config_delimiter__):]
            else:
                field_name = field.name

            # Nested config need to be treated differently, propagating kwargs if necessary
            field_type = type_hints[field.name]
            if isinstance(type_hints[field_name], type) and issubclass(field_type, BaseSparkConfig):
                # Attribute is another SparkConfig
                init_args = kwargs_fold.get(field_name, shared_partial)
                if field.default_factory is not dataclasses.MISSING:
                    # Use default factory if provided
                    setattr(self, field_name, field.default_factory(**init_args))
                else:
                    # Use type class otherwise
                    setattr(self, field_name, field_type(**init_args))
            elif field_name in kwargs_fold:
                # Use kwargs attribute if provided
                setattr(self, field_name, kwargs_fold[field_name])
            elif field.default_factory is not dataclasses.MISSING:
                # Fallback to default factory.
                setattr(self, field_name, field.default_factory())
            elif field.default is not dataclasses.MISSING:
                # Fallback to default.
                setattr(self, field_name, field.default)
            else:
                # TODO: Throw a better error than the default dataclass error.
                pass

    @classmethod
    def _create_partial(cls):
        """
            Create an incomplete config for the SparkGraphEditor.
        """
        # Manually create an instance
        instance = cls.__new__(cls)

        # Get type hints
        type_hints = tp.get_type_hints(instance.__class__)
        for key in type_hints.keys():
            if tp.get_origin(type_hints[key]): 
                hints = list(tp.get_args(type_hints[key]))
                type_hints[key] = (h for h in hints if h is not type(None))

        # Set attributes programatically
        for field in dataclasses.fields(instance):

            # Nested config need to be treated differently, propagating kwargs if necessary
            field_type = type_hints[field.name]
            if isinstance(type_hints[field.name], type) and issubclass(field_type, BaseSparkConfig):
                # Field is another SparkConfig use create partial recursively.
                setattr(instance, field.name, field_type._create_partial())
            elif field.default_factory is not dataclasses.MISSING:
                # Fallback to default factory.
                setattr(instance, field.name, field.default_factory())
            elif field.default is not dataclasses.MISSING:
                # Fallback to default.
                setattr(instance, field.name, field.default)
            else:
                setattr(instance, field.name, None)
                
        return instance
            
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
    def _fold_partial(obj: 'BaseSparkConfig', partial: dict[str, Any]) -> tuple(dict[str, Any], dict[str, Any]):
        fold_partial = {}
        shared_partial = {}
        for key, value in partial.items():

            # Check for specific subconfig attributes
            if obj.__config_delimiter__ in key:
                # Split key into child and nested_key
                child_key, nested_key = key.split(obj.__config_delimiter__, maxsplit=1)
                # Add nested key to child dict
                if child_key not in fold_partial:
                    fold_partial[child_key] = {}
                fold_partial[child_key][nested_key] = value

            # Check for shared attributes
            elif obj.__shared_config_delimiter__ == key[:len(obj.__shared_config_delimiter__)]:
                shared_key = key[len(obj.__shared_config_delimiter__):]
                shared_partial[shared_key] = value

            # Attribute is part of the current level
            else:
                fold_partial[key] = value

            # Add shared keys to current level and all its childs
            for s_key, value in shared_partial.items():
                fold_partial[s_key] = value
            for key in fold_partial.keys():
                if isinstance(fold_partial[key], dict):
                    for s_key, value in shared_partial.items():
                        # Add __shared_config_delimiter__ to allow for further propagation of the parameter
                        fold_partial[key][obj.__shared_config_delimiter__ + s_key] = value
        
        return fold_partial, shared_partial
    
    @staticmethod
    def _set_partial_attributes(obj: 'BaseSparkConfig', partial: dict[str, Any]) -> None:

        # Get type hints
        type_hints = tp.get_type_hints(obj.__class__)
        for key in type_hints.keys():
            if tp.get_origin(type_hints[key]): 
                hints = list(tp.get_args(type_hints[key]))
                type_hints[key] = (h for h in hints if h is not type(None))

        # Get object fields
        valid_fields = {field.name: field for field in dataclasses.fields(obj) }

        # Replace field values if present in partial
        for key, value in partial.items():
            if key in valid_fields:
                if isinstance(type_hints[key], type) and issubclass(type_hints[key], BaseSparkConfig):
                    # Field is another SparkConfig.
                    child_config = type_hints[key].create(value)
                    setattr(obj, key, child_config)
                else:
                    # Field is a plain value.
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
        # Get type hints
        type_hints = tp.get_type_hints(self.__class__)
        for key in type_hints.keys():
            if tp.get_origin(type_hints[key]): 
                hints = list(tp.get_args(type_hints[key]))
                type_hints[key] = (h for h in hints if h is not type(None))

        # Collect all fields and map into a dict
        dataclass_dict = {}
        for field in dataclasses.fields(self):
            # Nested config need to be treated differently, propagating kwargs if necessary
            if isinstance(type_hints[field.name], type) and issubclass(type_hints[field.name], BaseSparkConfig):
                dataclass_dict[field.name] = getattr(self, field.name).to_dict()
            else:
                dataclass_dict[field.name] = getattr(self, field.name)

        return dataclass_dict

    def _get_nested_configs_names(self,):
        """
            Returns a list containing all nested SparkConfigs' names.
        """
        # Get type hints
        type_hints = tp.get_type_hints(self.__class__)
        for key in type_hints.keys():
            if tp.get_origin(type_hints[key]): 
                hints = list(tp.get_args(type_hints[key]))
                type_hints[key] = (h for h in hints if h is not type(None))

        # Collect all fields and map into a dict
        nested_configs = []
        for field in dataclasses.fields(self):
            # Check if field is another SparkConfig
            if isinstance(type_hints[field.name], type) and issubclass(type_hints[field.name], BaseSparkConfig):
                nested_configs.append(field.name)

        return nested_configs

    def _freeze(self,) -> FrozenSparkConfig:
        """
            Returns a frozen version of the config instance. 
            Use to make the SparkConfig compatible with JIT.
        """
        return FrozenSparkConfig(self)

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