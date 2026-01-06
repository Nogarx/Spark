#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

#from __future__ import annotations

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
import spark.core.utils as utils
from jax.typing import DTypeLike
from spark.core.validation import(
    _is_config_instance, _is_config_type, _is_module_instance,
    _is_initializer_config_type, _is_initializer_type, _is_initializer_instance
)
from spark.core.registry import REGISTRY, register_config
from spark.core.config_validation import TypeValidator, PositiveValidator
from spark.core.signature_parser import normalize_typehint, is_instance
from functools import partial, wraps
from math import prod

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

METADATA_TEMPLATE = {
    'units': None, 
    'valid_types': tp.Any, 
    'validators': [
        TypeValidator,
    ], 
    'description': '',
    'allows_init': False,
}
IMMUTABLE_TYPES = (str, int, bool, float, tuple, type(None))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InitializableFieldMetaclass(type):
    """
        Metaclass that automatically injects common methods into the class.
    """

    def __new__(mcs, name, bases, attrs) -> tp.Self:

        DELEGATED_METHODS = [
            # Container Methods
            '__len__', '__getitem__', '__setitem__', '__delitem__', '__contains__',
            '__iter__', '__reversed__',
            # Arithmetic/Comparison
            '__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', 
            '__truediv__', '__rtruediv__', '__floordiv__', '__rfloordiv__', '__mod__', 
            '__rmod__', '__pow__', '__rpow__',
            '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__',
            # Unary operators
            '__neg__', '__pos__', '__abs__',
            # Callable and Context Manager
            '__call__', '__enter__', '__exit__',
        ]

        def _delegate_method_factory(method_name):
            """Creates a function that delegates a dunder method call to the wrapped object."""
            def delegator(self, *args, **kwargs):
                # Forward method to __obj__.
                return getattr(self.__obj__, method_name)(*args, **kwargs)
            # Forward metadata from a potential base method.
            return wraps(getattr(object, method_name, lambda *a, **k: None))(delegator)

        # Inject methods into the class dictionary
        for method_name in DELEGATED_METHODS:
            if method_name not in attrs:
                attrs[method_name] = _delegate_method_factory(method_name)
        return super().__new__(mcs, name, bases, attrs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InitializableField(metaclass=InitializableFieldMetaclass):
    """
        Wrapper for fields that allow for Initializers | InitializersConfig to define the init() method.
        The method init() is extensively used through Spark modules to initialize variables either from default
        values or from full fledge initializers.
    """
    __obj__: tp.Any

    def __init__(self, obj) -> None:
        # Avoid wrapping an already wrapped object.
        if isinstance(obj, InitializableField):
            obj = obj.__obj__
        super().__setattr__('__obj__', obj)

    def __getattr__(self, name) -> tp.Any:
        if name == '__obj__':
            return super().__getattr__('__obj__')
        else:
            return getattr(self.__obj__, name)

    def __setattr__(self, name, value) -> None:
        setattr(self.__obj__, name, value)

    def __repr__(self) -> str:
        return repr(self.__obj__)

    def __str__(self) -> str:
        return str(self.__obj__)

    def init(
            self, 
            init_kwargs={}, 
            key: jax.Array | None = None, 
            shape: tuple | None = None, 
            dtype: np.dtype | None = None,
            **kwargs
        ) -> jax.Array | int | float | complex | bool:
        from spark.nn.initializers import Initializer, InitializerConfig
        if isinstance(self.__obj__, Initializer):
            return self.__obj__(key=key, shape=shape, **kwargs)
        elif isinstance(self.__obj__, InitializerConfig):
            init_args = {
                **self.__obj__.get_kwargs(),
                **init_kwargs,
                **({'dtype': dtype} if dtype is not None else {})
            }
            return self.__obj__.class_ref(**init_args)(key=key, shape=shape, **kwargs)
        elif isinstance(self.__obj__, (int, float, complex, bool, np.ndarray, jax.Array)):
            return self.__obj__
        else:
            raise TypeError(
                f'Expected __obj__ to be of type numeric | Initializer | InitializerConfig but got {self.__obj__}'
            )

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
                allows_init = 'Initializer' in attr_type or 'InitializerConfig' in attr_type
            elif tp.get_origin(attr_type):
                attr_types = list(tp.get_args(attr_type))
                try:
                    attr_types.remove(type(None))
                except:
                    pass
                allows_init = any([_is_initializer_type(ft) or _is_initializer_config_type(ft) for ft in attr_types])
                attr_types = '|'.join([ft.__name__ for ft in attr_types])
                valid_types = {'valid_types': attr_types}
            else:
                allows_init = _is_initializer_type(attr_type) or _is_initializer_config_type(attr_type)
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
                # Save as much as we can of the intention of default_factory
                factory = attr_value.default_factory
                if factory != dc.MISSING and (_is_config_instance(factory) or _is_initializer_instance(factory) or not callable(factory)):
                    raise TypeError(
                        f'Expected \"{name}.{attr_name}.default_factory\" to be an instance of a function ' \
                        f'or a subclass of SparkConfig | Initializer but got type {type(factory)}.' \
                        f'If you intended to use an instance of a SparkConfig or an initializer use \"default\" instead'
                    )

                default, factory = cls.map_common_init_patterns(attr_value.default, attr_value.default_factory)
                field = dc.field(
                    default=default,
                    default_factory=factory,
                    init=attr_value.init,
                    repr=attr_value.repr,
                    hash=attr_value.hash,
                    compare=attr_value.compare,
                    metadata={
                        **METADATA_TEMPLATE, 
                        **(attr_value.metadata or {}), 
                        **valid_types, 
                        **validators,
                        **{'allows_init': allows_init}
                    },
                )

            # If is not field, promote to field
            else:
                default, factory = cls.map_common_init_patterns(attr_value)
                # Create a new Field object.
                field = dc.field(
                    default=default,
                    default_factory=factory,
                    metadata={
                        **METADATA_TEMPLATE, 
                        **valid_types,
                        **{'allows_init': allows_init}
                    }
                )

            # Update dct
            dct[attr_name] = field

        # Create the dataclass
        new_class = super().__new__(cls, name, bases, dct)

        return new_class

    # TODO: Verify and clean code
    def map_common_init_patterns(default: tp.Any, factory: tp.Callable = dc.MISSING) -> tuple[tp.Any, tp.Any]:
        if default != dc.MISSING and isinstance(default, (list, dict, set, np.ndarray, jax.Array)):
            # Map list, arrays, dicts and sets to lambdas in order to be able to use them directly as defaults.
            return dc.MISSING, lambda value=default : value
        elif default != dc.MISSING and _is_config_instance(default):
            # Map already initialized configs to lambda 
            return dc.MISSING, lambda value=default : value
        elif default != dc.MISSING and _is_config_type(default):
            # Map configs types to lambdas 
            return dc.MISSING, default
        elif default != dc.MISSING and _is_initializer_instance(default):
            # Map already initialized initializers to lambda of its configs
            return dc.MISSING, lambda value=default.config : value
        elif default != dc.MISSING and _is_initializer_type(default):
            # Get config from initializer and map it to lambda of its config
            return dc.MISSING, lambda value=default.get_config_spec() : value()
        elif default != dc.MISSING and _is_module_instance(default):
            # Map already initialized modules to lambda of its config
            return dc.MISSING, lambda value=default.config : value
        elif default != dc.MISSING and callable(default) and not utils.is_dtype(default):
            # Default is a function
            return dc.MISSING, default
        elif factory != dc.MISSING and _is_config_type(factory):
            # Map configs types to lambdas 
            return dc.MISSING, lambda value=factory : value()
        elif factory != dc.MISSING and _is_initializer_type(factory):
            # Get config from initializer and map it to lambda of its config
            return dc.MISSING, lambda value=factory.get_config_spec() : value()
        else:
            return default, factory

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_dataclass
@dc.dataclass(init=False, kw_only=True, eq=False)
class BaseSparkConfig(abc.ABC, metaclass=SparkMetaConfig):
    """
        Base class for module configuration.
    """

    __config_delimiter__: str = '__'
    __shared_config_delimiter__: str = '_s_'
    __metadata__: dict = dc.field(default_factory = lambda: {})
    __graph_editor_metadata__: dict = dc.field(default_factory = lambda: {})



    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dc.dataclass(cls, init=False, kw_only=True, repr=False, eq=False)


    def __init__(self, __skip_validation__: bool = False, **kwargs):
        # Fold kwargs
        kwargs_fold, shared_partial = self._fold_partial(self, kwargs)
        # Set attributes programatically
        self._set_partial_attributes(kwargs_fold, shared_partial,  __skip_validation__=__skip_validation__)
        # TODO: __graph_editor_metadata__ is not being set automatically
        self.__graph_editor_metadata__ = kwargs['__graph_editor_metadata__'] if '__graph_editor_metadata__' in kwargs else {}
        self.__metadata__ = kwargs['__metadata__'] if '__metadata__' in kwargs else {}
        # Validate
        if not __skip_validation__:
            self.__post_init__()
        # Wrap attributes that admit initializers
        for field in dc.fields(self):
            if field.metadata['allows_init']:
                setattr(self, field.name, InitializableField(getattr(self, field.name)))

    def __eq__(self, other) -> bool:
        if not isinstance(other, BaseSparkConfig):
            return False
        from spark.nn.initializers import Initializer, InitializerConfig
        # Check both of them contain the same fields.
        fields = set([f.name for f in dc.fields(self)])
        other_fields = set([f.name for f in dc.fields(other)])
        if fields != other_fields:
            return False
        # Compare fields contents.
        for name in fields:
            value = getattr(self, name)
            other_value = getattr(self, name)
            if type(value) != type(other_value):
                return False
            if isinstance(value, BaseSparkConfig):
                # Recursive comparison.
                if value != other_value:
                    return False
            elif isinstance(value, InitializableField) and isinstance(value.__obj__, Initializer):
                if value.__obj__.config != other_value.__obj__.config:
                    return False
            elif isinstance(value, InitializableField):
                if value.__obj__ != other_value.__obj__:
                    return False
            elif isinstance(value, (jax.Array, np.ndarray)):
                if np.all(value != other_value):
                    return False
            else:
                if value != other_value:
                    return False
        return True

    @classmethod
    def _create_partial(cls, **kwargs) -> tp.Self:
        """
            Create an incomplete config for the SparkGraphEditor.
        """
        # Manually create an instance
        instance = cls.__new__(cls)
        # Fold kwargs
        kwargs_fold, shared_partial = instance._fold_partial(instance, kwargs)
        # Set attributes programatically
        instance._set_partial_attributes(kwargs_fold, shared_partial, True)
        # TODO: __graph_editor_metadata__ is not being set automatically
        instance.__graph_editor_metadata__ = kwargs['__graph_editor_metadata__'] if '__graph_editor_metadata__' in kwargs else {}
        instance.__metadata__ = kwargs['__metadata__'] if '__metadata__' in kwargs else {}
        # Wrap attributes that admit initializers
        for field in dc.fields(instance):
            if field.metadata['allows_init']:
                setattr(instance, field.name, InitializableField(getattr(instance, field.name)))
        return instance



    def merge(self, partial: dict[str, tp.Any] = {}, __skip_validation__: bool = False) -> None:
        """
            Update config with partial overrides.
        """
        # Fold partial to pass child config attributes.
        kwargs_fold, shared_partial = self._fold_partial(self, partial)
        # Get current fields and pass attributes.
        self._set_partial_attributes(kwargs_fold, shared_partial, __skip_validation__=__skip_validation__)
        # Validate
        if not __skip_validation__:
            self.__post_init__()
        # Wrap attributes that admit initializers
        for field in dc.fields(self):
            if field.metadata['allows_init']:
                setattr(self, field.name, InitializableField(getattr(self, field.name)))



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



    # TODO: We need to validate that whenever we set a config class it is a valid subclass for the field
    def _set_partial_attributes(
            self, 
            kwargs_fold: dict[str, tp.Any], 
            shared_partial: dict[str, tp.Any], 
            __skip_validation__: bool = False,
        ) -> None:
        """
            Method to recursively set/override the attributes of the configuration instance from a dictionary of values.
        """
        # Get type hints
        from spark.nn.initializers.base import Initializer, InitializerConfig
        type_hints = {k: normalize_typehint(v) for k,v in tp.get_type_hints(self.__class__).items()}
        optional_attrs = {k: type(None) in v for k,v in type_hints.items()}

        for k in type_hints.keys():
            if Initializer in type_hints[k] and not InitializerConfig in type_hints[k]:
                type_hints[k] += (InitializerConfig,)
            elif InitializerConfig in type_hints[k] and not Initializer in type_hints[k]:
                type_hints[k] += (Initializer,)

        # Set attributes programatically
        prefixed_shared_partial = {f'{self.__shared_config_delimiter__}{k}':v for k,v in shared_partial.items()}
        for field in dc.fields(self.__class__):

            # Skip metadata fields
            if field.name in [
                '__config_delimiter__', 
                '__shared_config_delimiter__', 
                '__class_ref__', 
                '__graph_editor_metadata__',
                '__metadata__',
            ]:
                continue
            
            # Parse field name, remove __shared_config_delimiter__ if present
            if self.__shared_config_delimiter__ == field.name[:len(self.__shared_config_delimiter__)]:
                field_name = field.name[len(self.__shared_config_delimiter__):]
            else:
                field_name = field.name

            # Check if attribute is optional and the user manually set it to None. Kwargs have priority over defaults.
            if optional_attrs[field_name] and (
                (field_name in kwargs_fold.keys() and isinstance(kwargs_fold[field_name], type(None))) or
                (not field_name in kwargs_fold.keys() and getattr(self, field_name, None) == None and isinstance(field.default, type(None)))
                ):
                # Unfortunately, the user is correct in this one, set the parameter to None and move on.
                field.default_factory = dc.MISSING
                field.default = None
                setattr(self, field_name, None)
                continue

            # Current field relevant values
            field_type = type_hints[field_name]
            field_value = getattr(self, field_name, None)

            # Unpack.
            # TODO: Packing and unpacking looks dirty, maybe there is a workaround?
            if isinstance(field.default, InitializableField):
                field.default = field.default.__obj__
            if isinstance(field.default_factory, InitializableField):
                field.default_factory = field.default_factory.__obj__
            if isinstance(field_value, InitializableField):
                field_value = field_value.__obj__

            # Fields of type Initializer are appended the class InitializerConfig.
            # We need to resolve simple fields and kwargs before the Config case 
            if any([isinstance(t, type) and issubclass(t, Initializer) for t in field_type]):

                if field_name in kwargs_fold:
                    if isinstance(kwargs_fold[field_name], Initializer):
                        # Extract config and let it reach the next section
                        kwargs_fold[field_name] = kwargs_fold[field_name].config
                    elif isinstance(kwargs_fold[field_name], InitializerConfig):
                        # Let it reach the next section
                        pass
                    elif not isinstance(kwargs_fold[field_name], dict):
                        # NOTE: dict means that want to initialize a config.
                        # Use kwargs attribute to set field
                        self._validate_and_set(field_name, kwargs_fold[field_name], field_type)
                        continue
                elif not isinstance(field_value, type(None)):
                    if isinstance(field_value, Initializer):
                        # Extract config and let it reach the next section
                        field_value = field_value.config
                    elif isinstance(field_value, InitializerConfig):
                        # Let it reach the next section
                        pass
                    else:
                        # Use field_value attribute to set field
                        self._validate_and_set(field_name, field_value, field_type)
                        continue
                else:
                    if len(field_type) > 2 and field.default == dc.MISSING and field.default_factory == dc.MISSING and field_value == None:
                        raise ValueError(
                            f'Field \"{self.__class__.__name__}.{field_name}\" does not define a default value and multiple types are ' \
                            f'provided, automatic resolution is not possible for fields with the type Initializer | InitializerConfig ' \
                            f'when multiple types are provided. Please define a default value for this field.'
                        )

            # Nested Config case
            if any([isinstance(t, type) and issubclass(t, BaseSparkConfig) for t in field_type]):
                # Check if user manually set this attribute and is a SparkConfig
                subconfig = kwargs_fold.get(field_name, {})
                if isinstance(subconfig, BaseSparkConfig):
                    # User set a config. Copy subconfig to avoid weird overwrittings.
                    field_config = copy.deepcopy(subconfig)
                    field_config.merge(partial={**prefixed_shared_partial}, __skip_validation__=__skip_validation__)
                    setattr(self, field_name, field_config)
                elif isinstance(subconfig, type) and issubclass(subconfig, BaseSparkConfig):
                    # Use current config 
                    field_value = subconfig(**{**subconfig, **prefixed_shared_partial}, __skip_validation__=__skip_validation__)
                    setattr(self, field_name, field_value)
                # Check if parent config set this attribute and is a SparkConfig
                elif isinstance(field_value, BaseSparkConfig):
                    # Use current config 
                    field_value.merge(partial={**subconfig, **prefixed_shared_partial}, __skip_validation__=__skip_validation__)
                    setattr(self, field_name, field_value)
                elif isinstance(field_value, type) and issubclass(field_value, BaseSparkConfig):
                    # Use current config 
                    field_value = field_value(**{**subconfig, **prefixed_shared_partial}, __skip_validation__=__skip_validation__)
                    setattr(self, field_name, field_value)
                # Use default factory if provided
                elif field.default_factory is not dc.MISSING:
                    # NOTE: This case is not only covering generic factories, but also the case when a SparkConfig() 
                    # is used as a default value. The simple solution is call the factory, then merge with default parameters.
                    field_value = field.default_factory()
                    field_value.merge(
                        partial={**subconfig, **prefixed_shared_partial},
                        __skip_validation__=__skip_validation__
                    )
                    setattr(self, field_name, field_value)
                # Use type class otherwise
                else:
                    # TODO: It is not clear how to fully resolve multiple BaseSparkConfig types. We currently select the first one.
                    valid_types = [isinstance(t, type) and issubclass(t, BaseSparkConfig) for t in field_type]
                    if sum(valid_types) > 1:
                        raise TypeError(
                            f'Multiple SparkConfig types for field \"{field_name}\" in \"{self.__class__.__name__}\" were assigned. '\
                            f'It is not possible to automatically resolve config initialization. ' \
                            f'Only one SparkConfig class per field annotation can be used to allow for automatic initialization. ' \
                            f'If this is inteded, define a default_factory or provide a SparkConfig to the parent initializer. '
                        )
                    # Recover instance
                    valid_field_type = field_type[valid_types.index(True)]
                    setattr(self, field_name, valid_field_type(**{
                        **subconfig, 
                        **prefixed_shared_partial,
                        **{'__skip_validation__': __skip_validation__}
                    }))
                continue
            
            # Generic case
            if field_name in kwargs_fold:
                # Use kwargs attribute if provided
                self._validate_and_set(field_name, kwargs_fold[field_name], field_type)
                continue
            elif not isinstance(field_value, type(None)):
                # Use field_value attribute if provided
                self._validate_and_set(field_name, field_value, field_type)
                continue
            elif isinstance(field_value, type(None)) and field.default_factory is not dc.MISSING:
                # Fallback to default factory.
                setattr(self, field_name, field.default_factory())
                continue
            elif isinstance(field_value, type(None)) and field.default is not dc.MISSING:
                # Fallback to default.
                setattr(self, field_name, field.default)
                continue
            else:
                if __skip_validation__:
                    continue
                # Try a default initialization of the parameter
                if len(field_type) > 1:
                    raise ValueError(
                        f'Field "{field_name}" does not define a default value and multiple types are provided, ' \
                        f'automatic resolution is not possible. ' \
                        f'Please define a default value for this field.'
                    )
                else:
                    try:
                        default_value = field_type[0]()
                        field.default = default_value
                        setattr(self, field_name, default_value)
                        continue
                    except:
                        raise ValueError(
                            f'Field "{field_name}" does not define a default value, ' \
                            f'and automatic resolution failed to produce a default value. ' \
                            f'Please define a default value for this field.'
                        )



    def _validate_and_set(self, field_name, value, field_type):
        # Use kwargs attribute if provided
        if is_instance(value, field_type):
            # kwargs_fold provides a valid type:
            setattr(self, field_name, value)
        else:
            # Try some safe promotions
            valid_safer_types = [float, int, bool, str, tuple, list]
            for t in field_type:
                origin = tp.get_origin(t) or t
                if origin in valid_safer_types:
                    try:
                        safe_value = origin(value)
                        setattr(self, field_name, safe_value)
                        return
                    except:
                        pass
            raise TypeError(
                f'Field {field_name} got value of type {type(value)} but ' \
                f'it is not possible to safely promote it to the types: {field_type}.'
            )



    def diff(self, other: 'BaseSparkConfig') -> dict[str, tp.Any]:
        """
            Return differences from another config.
        """
        return {
            field.name: getattr(self, field.name) 
            for field in dc.fields(self) 
            if getattr(self, field.name) != getattr(other, field.name)
        }



    def _get_nested_configs_names(self,) -> list[tuple[type,...]]:
        """
            Returns a list containing all nested SparkConfigs' names.
        """
        # Get type hints
        type_hints = tp.get_type_hints(self.__class__)
        for key in type_hints.keys():
            if tp.get_origin(type_hints[key]): 
                hints = list(tp.get_args(type_hints[key]))
                type_hints[key] = tuple([h for h in hints if h is not type(None)])
            else:
                type_hints[key] = tuple([type_hints[key]])

        # Collect all fields and map into a dict
        nested_configs = []
        for field in dc.fields(self):
            # Check if field is another SparkConfig
            if any([isinstance(t, type) and issubclass(t, BaseSparkConfig) for t in type_hints[field.name]]):
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
        # Get type hints
        raw_type_hints = tp.get_type_hints(self.__class__)
        optional_attrs = {}
        for key in raw_type_hints.keys():
            if tp.get_origin(raw_type_hints[key]): 
                hints = list(tp.get_args(raw_type_hints[key]))
                optional_attrs[key] = len(tuple([h for h in hints if h is type(None)])) > 0
            else:
                optional_attrs[key] = False
        # Iterate over fields
        for field in dc.fields(self):
            # Skip optional fields
            if optional_attrs[field.name] and (getattr(self, field.name, None) != None):
                continue
            # Validate field
            for validator in field.metadata.get('validators', []):
                validator_instance = validator(field)
                value_attr = getattr(self, field.name)
                # Remove descriptor
                if isinstance(value_attr, InitializableField):
                    value_attr = value_attr.__obj__
                # TODO: Add a wrapper to init to perform dynamic validation
                if isinstance(value_attr, BaseSparkConfig):
                    continue
                validator_instance.validate(value_attr)



    def get_field_errors(self, field_name: str) -> list[str]:
        """
            Validates all fields in the configuration class.
        """
        # Validate field
        field: dc.Field = getattr(self, '__dataclass_fields__').get(field_name, None)
        if field is None:
            raise KeyError(
                f'SparkConfig \"{self.__class__}\" does not define a field with name \"{field_name}\".'
            )
        # Try to validate field
        errors = []
        for validator in field.metadata.get('validators', []):
            validator_instance = validator(field)
            try:
                validator_instance.validate(getattr(self, field.name))
            except Exception as e:
                errors.append(f'{validator.__name__}: {e}')
        return errors



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
            when defining custom configuration classes. The automatic class_ref solver is extremely brittle and
            likely to fail in many different custom scenarios.
        """
        # Check if is a BrainConfig
        from spark.nn.brain import Brain, BrainConfig
        if isinstance(obj, BrainConfig):
            return Brain

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
                f'Configuration \"{obj.__class__.__name__}\" cannot resolve __class_ref__. '
                f'A Module and an Initializer with the same reference were found. '
                f'To prevent errors impute the class manually. Alternatively, update the name '
                f'of one of the classes to avoid overlappings.'
            )
        if module_class_ref:
            class_ref = module_class_ref.class_ref
        elif initializer_class_ref: 
            class_ref = initializer_class_ref.class_ref
        else:
            raise AttributeError(
                f'Configuration \"{obj.__class__.__name__}\" cannot resolve __class_ref__. '
                f'No Module nor Initializer with the same reference were found. '
                f'Either rename the configuration object as \"Object.__class__.__name__ + Config\" or'
                f'manually define __class_ref__ using the registry name of the object (default: Object.__class__.__name__).'
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
        from spark.nn.initializers import Initializer, InitializerConfig
        # Collect all fields and map into a dict
        dataclass_dict = {}
        for field in dc.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, BaseSparkConfig):
                dataclass_dict[field.name] = value
            elif isinstance(value, InitializableField) and isinstance(value.__obj__, Initializer):
                dataclass_dict[field.name] = value.__obj__.config
            elif isinstance(value, InitializableField) and isinstance(value.__obj__, InitializerConfig):
                dataclass_dict[field.name] = value.__obj__
            elif isinstance(value, InitializableField):
                dataclass_dict[field.name] = value.__obj__
            else:
                dataclass_dict[field.name] = value
        return dataclass_dict



    def get_kwargs(self) -> dict[str, dict[str, tp.Any]]:
        """
            Returns a dictionary with pairs of key, value fields (skips metadata).
        """
        from spark.nn.initializers import Initializer, InitializerConfig
        kwargs_dict = {}
        for name, field, value in self:
            if isinstance(value, BaseSparkConfig):
                kwargs_dict[name] = value.get_kwargs()
            elif isinstance(value, InitializableField):
                kwargs_dict[name] = value.__obj__
            else:
                kwargs_dict[name] = value
        return kwargs_dict



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
                        f'Reconstruction from unregistered classes is not currently possible.'
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
    def from_file(cls: type['BaseSparkConfig'], file_path: str) -> 'BaseSparkConfig':
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
                except Exception as e:
                    raise RuntimeError(
                        f'An unexpected error ocurred when trying to decode... '
                        f'418 I\'m a teapot.'
                        f'Error: {e}'
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
            if f.name in [
                '__config_delimiter__', 
                '__shared_config_delimiter__', 
                '__class_ref__', 
                '__graph_editor_metadata__',
                '__metadata__'
            ]:
                continue

            # Yield the field name and its corresponding value
            yield (f.name, f, getattr(self, f.name, None))



    def __repr__(self,):
        return f'{self.__class__.__name__}(...)'



    def inspect(self, simplified=False) -> str:
        """
            Returns a formated string of the datastructure.
        """
        print(utils.ascii_tree(self._parse_tree_structure(0, simplified=simplified)))



    def _inspect(self, simplified=True) -> str:
        """
            Returns a formated string of the datastructure.
        """
        return utils.ascii_tree(self._parse_tree_structure(0, simplified=simplified))



    def _parse_tree_structure(self, current_depth: int, simplified: bool = False, header: str | None= None) -> str:
        """
            Parses the tree to produce a string with the appropiate format for the ascii_tree method.
        """
        level_header = f'{header}: ' if header else ''
        rep = current_depth * ' ' + f'{level_header}{self.__class__.__name__}\n'
        for name, field, value in self:
            if not simplified:
                if isinstance(value, InitializableField):
                    # Unpack Initializable fields
                    value = value.__obj__
                if isinstance(value, BaseSparkConfig):
                    rep += value._parse_tree_structure(current_depth+1, simplified=simplified, header=name)
                else:
                    if isinstance(value, (list, tuple, set)) and len(value) > 5:
                        value_str = str(value_str[:5])
                        value_str = f'{type(value)}([{value_str[1:-1]}, ...])'
                    elif isinstance(value, (np.ndarray, jnp.ndarray)) and prod(value.shape) > 5:
                        value_str = ', '.join([f'{x:.2f}'.rstrip('0').rstrip('.') for x in value.reshape(-1)[:5]]).strip('\n').replace('\n', '')
                        value_str = f'array([{value_str[:-1]}, ...], dtype={value.dtype})'
                    else:
                        value_str = str(value).strip('\n')

                    if field.type == jax.typing.DTypeLike:
                        field_types = 'DTypeLike'
                    else:
                        field_types = str(field.type)
                    #field_types = ' | '.join([t.replace(' ', '').split('.')[-1] for t in str(field.type).split('|')])
                    rep += (current_depth+1) * ' ' + f'{name}: {field_types} <- {value_str}\n'
            else:
                if isinstance(value, BaseSparkConfig):
                    rep += value._parse_tree_structure(current_depth+1, simplified=simplified)
        return rep

    def with_new_seeds(self, seed=None) -> 'BaseSparkConfig':
        """
            Utility method to recompute all seed variables within the SparkConfig.
            Useful when creating several populations from the same config.
        """
        new_instance = copy.deepcopy(self)
        if seed is None:
            for name, field, value in new_instance:
                if isinstance(value, BaseSparkConfig):
                    setattr(new_instance, name, value.with_new_seeds())
                elif name == 'seed':
                    setattr(new_instance, name, int.from_bytes(os.urandom(4), 'little'))
        else:
            key = jax.random.key(seed)
            for name, field, value in new_instance:
                key, subkey = jax.random.split(key, 2)
                new_seed = int(subkey._base_array[0])
                if isinstance(value, BaseSparkConfig):
                    setattr(new_instance, name, value.with_new_seeds(seed=new_seed))
                elif name == 'seed':
                    setattr(new_instance, name, new_seed)
        return new_instance

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

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