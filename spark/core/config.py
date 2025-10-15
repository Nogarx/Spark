#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import abc
import jax
import jax.numpy as jnp
import numpy as np
import dataclasses as dc
import typing as tp
import copy
import json
import pathlib as pl
from jax.typing import DTypeLike
from spark.core.validation import _is_config_instance
from spark.core.registry import REGISTRY, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# TODO: Add some logic to the metadata for auto-inference of certain parameters when using GUI.
# For example, some Shapes can be infered from the next/prev component in the graph.
METADATA_TEMPLATE = {
    'units': None, 
    'valid_types': tp.Any, 
    'validators': [], 
    'description': '',
}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkMetaConfig(abc.ABCMeta):
    """
        Metaclass that promotes class attributes to dataclass fields
    """
    
    def __new__(cls, name: str, bases: tuple[type, ...], dct: dict[str, tp.Any]) -> 'SparkMetaConfig':

        # NOTE: Every non field is promoted to field to simplify the logic of configuration objects and add metadata.
        
        # Iterate over annotations.
        annotations: dict[str, tp.Any] = dct.get('__annotations__', {})
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
            attr_value = dct.get(attr_name, dc.MISSING)

            # If is field, add template info
            if isinstance(attr_value, dc.Field):
                # Merge metadata with template.
                validators = {
                    'validators': attr_value.metadata['validators'] 
                    if 'validators' in attr_value.metadata 
                    else METADATA_TEMPLATE['validators']
                }
                field = dc.field(
                    default=attr_value.default,
                    default_factory=attr_value.default_factory,
                    init=attr_value.init,
                    repr=attr_value.repr,
                    hash=attr_value.hash,
                    compare=attr_value.compare,
                    metadata={
                        **METADATA_TEMPLATE, 
                        **(attr_value.metadata or {}), 
                        **valid_types, 
                        **validators
                    },
                )

            # If is not field, promote to field
            else:
                # Create a new Field object.
                field = dc.field(
                    default=attr_value,
                    metadata={
                        **METADATA_TEMPLATE, 
                        **valid_types
                    }
                )

            # Update dct
            dct[attr_name] = field

        # Create the dataclass
        new_class = super().__new__(cls, name, bases, dct)

        return new_class

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_dataclass
@dc.dataclass(init=False, kw_only=True)
class BaseSparkConfig(abc.ABC, metaclass=SparkMetaConfig):
    """
        Base class for module configuration.
    """

    __config_delimiter__: str = '__'
    __shared_config_delimiter__: str = '_s_'
    __graph_editor_metadata__ = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dc.dataclass(cls, init=False, kw_only=True)

    def __init__(self, **kwargs):
        # Fold kwargs
        kwargs_fold, shared_partial = self._fold_partial(self, kwargs)
        # Set attributes programatically
        self._set_partial_attributes(kwargs_fold, shared_partial)
        self.__post_init__()

    @classmethod
    def _create_partial(cls, **kwargs):
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
        for field in dc.fields(instance):

            # Nested config need to be treated differently, propagating kwargs if necessary
            field_type = type_hints[field.name]
            if isinstance(type_hints[field.name], type) and issubclass(field_type, BaseSparkConfig):
                # Field is another SparkConfig use create partial recursively.
                try:
                    factory = field.default_factory
                    if not factory or factory is dc.MISSING:
                        raise ValueError(
                            f'Configuration class does not define a default_factory for Subconfig: \"{field.name}\".'
                        )
                    is_config_cls = isinstance(factory, type) and issubclass(factory, BaseSparkConfig)
                    is_callable = callable(factory)
                    if not (is_callable or is_config_cls):
                        raise TypeError(
                            f'Configuration class defines default_factory for Subconfig: \"{field.name}\" '
                            f'expected factory of type \"{BaseSparkConfig.__name__} | tp.Callable" but got {factory}.'
                        )
                    if is_config_cls:
                        field_config = factory._create_partial(**kwargs)
                    else:
                        field_config = factory(**kwargs)
                    if not isinstance(field_config, BaseSparkConfig):
                        raise TypeError(
                            f'Expected factory output to be of type \"{BaseSparkConfig.__name__}\", but got \"{field_config.__class__}\".'
                        )
                    setattr(instance, field.name, field_config)
                except:
                    setattr(instance, field.name, field_type._create_partial(**kwargs))
            elif field.default_factory is not dc.MISSING:
                # Fallback to default factory.
                setattr(instance, field.name, field.default_factory())
            elif field.default is not dc.MISSING:
                # Fallback to default.
                setattr(instance, field.name, field.default)
            else:
                setattr(instance, field.name, None)
                
        # Set existing attributes
        kwargs_fold, shared_partial = instance._fold_partial(instance, kwargs)
        # Set attributes programatically
        instance._set_partial_attributes(kwargs_fold, shared_partial)
        return instance



    def merge(self, partial: dict[str, tp.Any] = {}) -> None:
        """
            Update config with partial overrides.
        """
        # Fold partial to pass child config attributes.
        kwargs_fold, shared_partial = self._fold_partial(self, partial)
        # Get current fields and pass attributes.
        self._set_partial_attributes(kwargs_fold, shared_partial)
        # Validate
        self.validate()



    @staticmethod
    def _fold_partial(obj: 'BaseSparkConfig', partial: dict[str, tp.Any]) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """
            Restructures the partial dict, consuming the __config_delimiter__ and __shared_config_delimiter__ 
            tokens and propagating the information within.
        """
        fold_partial = {}
        shared_partial = {}
        for key, value in partial.items():

            # Check for shared attributes
            if obj.__shared_config_delimiter__ == key[:len(obj.__shared_config_delimiter__)]:
                shared_key = key[len(obj.__shared_config_delimiter__):]
                shared_partial[shared_key] = value

            # Check for specific subconfig attributes
            elif obj.__config_delimiter__ in key:
                # Split key into child and nested_key
                child_key, nested_key = key.split(obj.__config_delimiter__, maxsplit=1)
                # Add nested key to child dict
                if child_key not in fold_partial:
                    fold_partial[child_key] = {}
                fold_partial[child_key][nested_key] = value

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



    def _set_partial_attributes(self, kwargs_fold: dict[str, tp.Any], shared_partial: dict[str, tp.Any]) -> None:
        """
            Method to recursively set/override the attributes of the configuration instance from a dictionary of values.
        """
        # Get type hints
        type_hints = tp.get_type_hints(self.__class__)
        for key in type_hints.keys():
            if tp.get_origin(type_hints[key]): 
                hints = list(tp.get_args(type_hints[key]))
                type_hints[key] = (h for h in hints if h is not type(None))

        # Set attributes programatically
        prefixed_shared_partial = {f'{self.__shared_config_delimiter__}{k}':v for k,v in shared_partial.items()}
        for field in dc.fields(self.__class__):

            # Parse field name, remove __shared_config_delimiter__ if present
            if self.__shared_config_delimiter__ == field.name[:len(self.__shared_config_delimiter__)]:
                field_name = field.name[len(self.__shared_config_delimiter__):]
            else:
                field_name = field.name

            # Nested config need to be treated differently, propagating kwargs if necessary
            field_type = type_hints[field_name]
            field_value = getattr(self, field_name, None)
            if isinstance(field_type, type) and issubclass(field_type, BaseSparkConfig):
                # Attribute is another SparkConfig
                subconfig = kwargs_fold.get(field_name, {})
                if isinstance(field_value, BaseSparkConfig):
                    # Use current config 
                    field_value.merge(partial={**subconfig, **prefixed_shared_partial})
                    setattr(self, field_name, field_value)
                elif isinstance(subconfig, BaseSparkConfig):
                    # Argument is a config. Copy subconfig to avoid weird overwrittings.
                    field_config = copy.deepcopy(subconfig)
                    field_config.merge(partial={**prefixed_shared_partial})
                    setattr(self, field_name, field_config)
                elif field.default_factory is not dc.MISSING:
                    # Use default factory if provided
                    print(field_name)
                    setattr(self, field_name, field.default_factory(**{**subconfig, **prefixed_shared_partial}))
                else:
                    # Use type class otherwise
                    setattr(self, field_name, field_type(**{**subconfig, **prefixed_shared_partial}))
            elif field_name in kwargs_fold:
                # Use kwargs attribute if provided
                setattr(self, field_name, kwargs_fold[field_name])
            elif isinstance(field_value, type(None)) and field.default_factory is not dc.MISSING:
                # Fallback to default factory.
                setattr(self, field_name, field.default_factory())
            elif isinstance(field_value, type(None)) and field.default is not dc.MISSING:
                # Fallback to default.
                setattr(self, field_name, field.default)
            else:
                # TODO: Throw a better error than the default dataclass error.
                pass



    def diff(self, other: 'BaseSparkConfig') -> dict[str, tp.Any]:
        """
            Return differences from another config.
        """
        return {
            field.name: getattr(self, field.name) 
            for field in dc.fields(self) 
            if getattr(self, field.name) != getattr(other, field.name)
        }



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
        for field in dc.fields(self):
            # Check if field is another SparkConfig
            if isinstance(type_hints[field.name], type) and issubclass(type_hints[field.name], BaseSparkConfig):
                nested_configs.append(field.name)

        return nested_configs



    def _get_type_hints(self,):
        """
            Returns a dict containing all type hints.
        """
        # Get type hints
        type_hints = tp.get_type_hints(self.__class__)
        for key in type_hints.keys():
            if tp.get_origin(type_hints[key]): 
                hints = list(tp.get_args(type_hints[key]))
                process_hints = tuple([h for h in hints if h is not type(None)])
                if np.dtype in process_hints or jnp.dtype in process_hints:
                    type_hints[key] = (np.dtype,)
                else:
                    type_hints[key] = process_hints
            else:
                type_hints[key] = (type_hints[key],)
        return type_hints
    


    def validate(self,) -> None:
        """
            Validates all fields in the configuration class.
        """
        for field in dc.fields(self):
            for validator in field.metadata.get('validators', []):
                validator_instance = validator(field)
                validator_instance.validate(getattr(self, field.name))



    def get_metadata(self) -> dict[str, tp.Any]:
        """
            Returns all the metadata in the configuration class, indexed by the attribute name.
        """
        metadata = {}
        for field in dc.fields(self):
            metadata[field.name] = dict(field.metadata)
        return metadata


    # TODO: This method is not ideal. It solves the module association problem in a very brittle way. 
    # There should be another better pattern for this problem.
    @property
    def class_ref(obj: 'BaseSparkConfig') -> type:
        """
            Returns the type of the associated Module/Initializer.

            NOTE: It is recommended to set the __class_ref__ to the name of the associated module/initializer
            when defining custom configuration classes. The current class_ref solver is extremely brittle and
            likely to fail in many different custom scenarios.
        """
        # Check for class_ref otherwise try to set it up.
        if getattr(obj, '__class_ref__', None) is None:
            if obj.__class__.__name__[-6:].lower() == 'config':
                obj.__class_ref__ = obj.__class__.__name__[:-6]
            else:
                # Config is not following convention, manual input of __class_ref__ is required.
                raise AttributeError(f'Configuration \"{obj.__name__}\" does not define a __class_ref__.')
        # Currently it can only be either a Module or a Initializer, so better check those two.
        module_class_ref = REGISTRY.MODULES.get(obj.__class_ref__)
        initializer_class_ref = REGISTRY.INITIALIZERS.get(obj.__class_ref__)
        # Check we only got one coincidence, otherwise throw an error to avoid headaches.
        if module_class_ref and initializer_class_ref:
            raise AttributeError(
                f'Configuration \"{obj.__name__}\" cannot resolve __class_ref__. '
                f'A Module and an Initializer with the same reference were found. '
                f'To prevent errors impute the class manually. Alternatively, update the name '
                f'of one of the classes to avoid overlappings (not recommended).'
            )
        if module_class_ref:
            class_ref = module_class_ref.class_ref
        elif initializer_class_ref: 
            class_ref = initializer_class_ref.class_ref
        else:
            raise AttributeError(
                f'Configuration \"{obj.__name__}\" cannot resolve __class_ref__. '
                f'No Module nor Initializer with the same reference were found. '
                f'Either rename the configuration object as \"Object.__name__ + Config\" or'
                f'manually define __class_ref__ using the registry name of the object (default: Object.__name__).'
            )
        return class_ref



    def __post_init__(self,):
        self.validate()
        # Get type hints
        type_hints = tp.get_type_hints(self.__class__)
        for key in type_hints.keys():
            if tp.get_origin(type_hints[key]): 
                hints = list(tp.get_args(type_hints[key]))
                type_hints[key] = (h for h in hints if h is not type(None))



    def to_dict(self) -> dict[str, dict[str, tp.Any]]:
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
        for field in dc.fields(self):
            # Nested config need to be treated differently, propagating kwargs if necessary
            if isinstance(type_hints[field.name], type) and issubclass(type_hints[field.name], BaseSparkConfig):
                dataclass_dict[field.name] = getattr(self, field.name).to_dict()
            else:
                dataclass_dict[field.name] = getattr(self, field.name)
        return dataclass_dict



    @classmethod
    def from_dict(cls: type['BaseSparkConfig'], dct: dict) -> 'BaseSparkConfig':
        """
            Create config instance from dictionary.
        """
        return cls(**dct)



    def to_file(self, file_path: str) -> None:
        """
            Export a config instance from a .scfg file.
        """
        try:
            # Validate path
            path = pl.Path(file_path)
            # Add suffix to file to clarify is a spark config.
            path = path.with_suffix('.scfg')
            # Ensure the parent directory exists.
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to file.
            from spark.core.serializer import SparkJSONEncoder
            with open(path, 'w', encoding='utf-8') as json_file:
                reg = REGISTRY.CONFIG.get_by_cls(self.__class__)
                if not reg:
                    raise RuntimeError(
                        f'Config class \"{self.__class__}\" is not in the registry.'
                        f'Use the \"register_config\" decorator to add the class to the registry.'
                    )
                # Add top config metadata
                json_dict = {
                    '__type__': reg.name,
                    '__cfg__': self.to_dict(),
                }
                json.dump(json_dict, json_file, cls=SparkJSONEncoder, indent=4)
            print(f'Successfully exported data to {path}')
        except Exception as e:
            raise Exception(
                f'ERROR: Could not write file \"{file_path}\". Reason: {e}'
            )



    @classmethod
    def from_file(cls: type['BaseSparkConfig'], file_path: str) -> None:
        """
            Create config instance from a .scfg file.
        """
        try:
            path = pl.Path(file_path)
            # Validate path
            if not path.is_file():
                raise FileNotFoundError(f'No file found at the specified path: \"{path}\".')
            # Parse the file
            with open(path, 'r', encoding='utf-8') as json_file:
                # Try to decode
                from spark.core.serializer import SparkJSONDecoder
                try:
                    obj = json.load(json_file, cls=SparkJSONDecoder) 
                except:
                    raise RuntimeError(
                        f'An unexpected error ocurred when trying to decode... '
                        f'418 I\'m a teapot.'
                    )
                if not _is_config_instance(obj):
                    raise TypeError(
                        f'Expected final object to be of type \"BaseSparkConfig\" but after decoding the final object was of type \"{obj.__class__}\".'
                    )
                return obj
            #return cls.from_dict(data)
        except Exception as e:
            raise Exception(
                f'ERROR: Could not read file \"{file_path}\". Reason: {e}'
            )



    def __iter__(self) -> tp.Iterator[tuple[str, dc.Field, tp.Any]]:
        """
            Custom iterator to simplify SparkConfig inspection across the entire ecosystem.
            This iterator excludes private fields.

            Output:
                field_name: str, field name 
                field_value: tp.Any, field value
        """
        # Iterate over all defined fields of the dataclass
        for f in dc.fields(self):
            # Check for fields to skip
            if f.name in ['__config_delimiter__', '__shared_config_delimiter__', '__class_ref__', '__graph_editor_metadata__']:
                continue

            # Yield the field name and its corresponding value
            yield (f.name, f, getattr(self, f.name))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class SparkConfig(BaseSparkConfig):
    """
        Default class for module configuration.
    """
    seed: int = dc.field(
        default_factory=lambda: int.from_bytes(os.urandom(4), 'little'), 
        metadata={
            'validators': [
                TypeValidator,
            ], 
            'description': 'Seed for internal random processes.',
        })
    dtype: DTypeLike = dc.field(
        default=jnp.float16, 
        metadata={
            'validators': [
                TypeValidator,
            ], 
            'value_options': [
                jnp.float16,
                jnp.float32,
            ],
            'description': 'Dtype used for JAX dtype promotions.',
        })
    dt: float = dc.field(
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