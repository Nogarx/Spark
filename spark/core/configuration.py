#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import abc
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import inspect
from jax.typing import DTypeLike
from typing import Any, Type
from functools import wraps
from dataclasses import fields, asdict, field, dataclass, Field

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Meta module to resolve metaclass conflicts
class SparkMetaConfig(type):
    """
        Metaclass that promotes class attributes to dataclass fields
    """
    def __new__(mcls, name: str, bases: tuple[type, ...], dct: dict[str, Any]) -> 'SparkMetaConfig':
        # Collect existing fields from base classes
        base_fields = {}
        for base in bases:
            if hasattr(base, '__dataclass_fields__'):
                base_fields.update(base.__dataclass_fields__)
        
        # Process class dictionary
        new_attrs = {}
        annotations = dct.get('__annotations__', {})
        fields = {}
        
        for attr_name, attr_value in dct.items():
            # Preserve special attributes and methods

            if attr_name.startswith('__') \
                or callable(attr_value) \
                or isinstance(attr_value, classmethod) \
                or isinstance(attr_value, staticmethod):
                new_attrs[attr_name] = attr_value
                continue
                
            # Preserve existing fields
            if isinstance(attr_value, Field):
                fields[attr_name] = attr_value
                continue
            
            # Promote simple attributes to fields
            field_meta = {}
            if hasattr(attr_value, '__metadata__'):
                field_meta = attr_value.__metadata__
            
            # Determine field type from annotation or value
            field_type = annotations.get(attr_name, type(attr_value))
            
            # Create field with default value
            fields[attr_name] = field(
                default=attr_value,
                metadata=field_meta
            )
            annotations[attr_name] = field_type
        
        # Update class dictionary with fields
        new_attrs.update(fields)
        if annotations:
            new_attrs['__annotations__'] = annotations
        
        # Create class as dataclass
        cls_instance = super().__new__(mcls, name, bases, new_attrs)
        return dataclass(cls_instance)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkConfig(metaclass=SparkMetaConfig):
    """
        Base class for module configuration.
    """
    __SCHEMA_VERSION__ = '1.0'
    seed: int | None = field(
        default=None, 
        metadata={
            'units': None,
            'valid_types': int | None,
            'validators': [
                (lambda x: (x is None) or isinstance(x, int), lambda x: f'seed must be of type None or Int, got "{x}".')
            ]
        })
    dtype: DTypeLike = field(
        default=jnp.float16, 
        metadata={
            'units': None,
            'valid_types': DTypeLike,
            'validators': [
                (lambda x: isinstance(jnp.dtype(x), jnp.dtype), lambda x: f'dtype must be a valid JAX DTypeLike, got "{x}".')
            ]
        })
    dt: float = field(
        default=1.0, 
        metadata={
            'units': 'ms',
            'valid_types': float,
            'validators': [
                (lambda x: isinstance(x, float), lambda x: f'dt must be of type float, got "{type(x)}".'),
                (lambda x: x > 0, lambda x: f'dt must be positive, got "{x}".'),
            ]
        })

    @classmethod
    def create(cls: Type['SparkConfig'], partial: dict[str, Any] = None) -> 'SparkConfig':
        """
            Create config with partial overrides.
        """
        # Get default instance
        instance = cls()
        # Apply partial updates
        if partial:
            valid_fields = {f.name for f in fields(cls)}
            for key, value in partial.items():
                if key in valid_fields:
                    setattr(instance, key, value)
                else:
                    raise ValueError(f"Invalid config key: {key}")
        return instance

    def merge(self, partial: dict[str, Any]) -> None:
        """
            Update config with partial overrides.
        """
        valid_fields = {f.name for f in fields(self) }
        for key, value in partial.items():
            if key in valid_fields:
                setattr(self, value)
            else:
                raise ValueError(f"Invalid config key: {key}")

    def diff(self, other: 'SparkConfig') -> dict[str, Any]:
        """
            Return differences from another config.
        """
        return {
            f.name: getattr(self, f.name) 
            for f in fields(self) 
            if getattr(self, f.name) != getattr(other, f.name)
        }

    @classmethod
    def from_dict(cls: Type['SparkConfig'], data: dict) -> 'SparkConfig':
        """
            Create config instance from dictionary
        """
        return cls(**data)
    
    def to_dict(self) -> dict:
        """
            Serialize config to dictionary
        """
        return {
            f.name: {'value': getattr(self, f.name),
                     'metadata': f.metadata} for f in fields(self) 
        }

    # Validation using field metadata
    def validate(self):
        for f in fields(self):
            for (validator, msg) in f.metadata.get('validators', []):
                if not validator(getattr(self, f.name)):
                    raise ValueError(f'Validation failed for "{f.name}". {msg(getattr(self, f.name))}')

    def __post_init__(self):
        self.validate()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################