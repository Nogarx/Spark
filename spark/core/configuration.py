#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import jax.numpy as jnp
import dataclasses
from jax.typing import DTypeLike
from typing import Any, Type, Callable
from collections import OrderedDict

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Meta module to resolve metaclass conflicts
class SparkMetaConfig(type):
    """
        Metaclass that promotes class attributes to dataclass fields
    """

    def __new__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> 'SparkMetaConfig':

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
                if (isinstance(attr_value, Callable) or isinstance(attr_value, property)) \
                    and attr_name not in attrs and not attr_name.startswith('__'):
                    attrs[attr_name] = attr_value

        # Gather fields from the current class. Prioritize child's attributes over parent's.
        current_annotations = attrs.get('__annotations__', {})
        for field_name, field_type in current_annotations.items():
            field_value = attrs.get(field_name, dataclasses.MISSING)
            fields[field_name] = (field_type, field_value)

        # --- Step 4: Process and standardize all fields with metadata ---
        processed_fields = OrderedDict()
        for field_name, (field_type, field_info) in fields.items():
            if isinstance(field_info, dataclasses.Field):
                # It's already a Field object. Reconstruct it with merged metadata.
                existing_metadata = field_info.metadata or {}
                default_metadata = {'units': None, 'valid_types': field_type, 'validators': []}
                final_metadata = {**default_metadata, **existing_metadata}
                
                # CORRECTED: Manually reconstruct the Field object instead of using replace().
                field_args = {
                    'default': field_info.default,
                    'default_factory': field_info.default_factory,
                    'init': field_info.init,
                    'repr': field_info.repr,
                    'hash': field_info.hash,
                    'compare': field_info.compare,
                    'metadata': final_metadata,
                }
                # kw_only is only available in Python 3.10+
                if hasattr(field_info, 'kw_only'):
                    field_args['kw_only'] = field_info.kw_only
                
                processed_field = dataclasses.field(**field_args)

            else:
                # It's a simple type hint or has a direct default value.
                # Create a new Field object from scratch.
                default_metadata = {'units': None, 'valid_types': field_type, 'validators': []}
                processed_field = dataclasses.field(
                    default=field_info if field_info is not dataclasses.MISSING else dataclasses.MISSING,
                    metadata=default_metadata
                )
            processed_fields[field_name] = (field_type, processed_field)

        # --- Step 5: Reorder the standardized fields ---
        non_default_fields = OrderedDict()
        default_fields = OrderedDict()

        for field_name, (field_type, field_obj) in processed_fields.items():
            # Now we can reliably check the Field object for defaults.
            if field_obj.default is dataclasses.MISSING and \
               field_obj.default_factory is dataclasses.MISSING:
                non_default_fields[field_name] = (field_type, field_obj)
            else:
                default_fields[field_name] = (field_type, field_obj)
        
        # --- Step 6: Rebuild attributes for the new class ---
        final_annotations = OrderedDict()
        
        for field_name, (field_type, field_obj) in non_default_fields.items():
            final_annotations[field_name] = field_type
            attrs[field_name] = field_obj

        for field_name, (field_type, field_obj) in default_fields.items():
            final_annotations[field_name] = field_type
            attrs[field_name] = field_obj

        attrs['__annotations__'] = final_annotations
        
        # --- Step 7: Create the class and apply the decorator ---
        new_class = super().__new__(cls, name, tuple([]), attrs)
        return dataclasses.dataclass(new_class)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkConfig(metaclass=SparkMetaConfig):
    """
        Base class for module configuration.
    """
    __SCHEMA_VERSION__ = '1.0'
    seed: int = dataclasses.field(
        default_factory=lambda: int.from_bytes(os.urandom(4), 'little'), 
        metadata={
            'units': None,
            'valid_types': int,
            'validators': [
                (lambda x: isinstance(x, int), lambda x: f'seed must be of type None or Int, got "{x}".')
            ]
        })
    dtype: DTypeLike = dataclasses.field(
        default=jnp.float16, 
        metadata={
            'units': None,
            'valid_types': DTypeLike,
            'validators': [
                (lambda x: isinstance(jnp.dtype(x), jnp.dtype), lambda x: f'dtype must be a valid JAX DTypeLike, got "{x}".')
            ]
        })
    dt: float = dataclasses.field(
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
            valid_fields = {field.name for field in dataclasses.fields(cls)}
            for key, value in partial.items():
                if key in valid_fields:
                    setattr(instance, key, value)
                else:
                    raise ValueError(f'Invalid config key: {key}')
        return instance

    def merge(self, partial: dict[str, Any]) -> None:
        """
            Update config with partial overrides.
        """
        valid_fields = {field.name for field in dataclasses.fields(self) }
        for key, value in partial.items():
            if key in valid_fields:
                setattr(self, key, value)
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
            field.name: {'value': getattr(self, field.name),
                     'metadata': field.metadata} for field in dataclasses.fields(self) 
        }

    # Validation using field metadata
    def validate(self):
        for field in dataclasses.fields(self):
            for (validator, msg) in field.metadata.get('validators', []):
                if not validator(getattr(self, field.name)):
                    raise ValueError(f'Validation failed for "{field.name}". {msg(getattr(self, field.name))}')

    def __post_init__(self):
        self.validate()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################