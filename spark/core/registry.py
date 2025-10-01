#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.module import SparkModule
    from spark.core.payloads import SparkPayload
    from spark.nn.initializers.base import Initializer
    from spark.core.config import BaseSparkConfig
    from spark.core.config_validation import ConfigurationValidator

import logging
import dataclasses as dc
import typing as tp
from collections.abc import Mapping, ItemsView
from spark.core.utils import normalize_name
import spark.core.validation as validation

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@dc.dataclass
class RegistryEntry:
    """
        Structured entry for the registry.
    """
    name: str
    class_ref: type
    path: list[str]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SubRegistry(Mapping):
    """
        Registry for registry_base_type.
    """

    def __init__(self, registry_base_type: str):
        self._raw_registry: dict[str, type[object]] = {}
        self._registry: dict[str, RegistryEntry] = {}
        self._leaf_class = set()
        self.__built__ = False
        self._registry_base_type = registry_base_type
        self._registry_base_type_name = registry_base_type.split('.')[-1]

    def __getitem__(self, key: str) -> RegistryEntry:
        if not self.__built__:
            raise RuntimeError('Registry is not build yet.')
        return self._registry[key]

    def __iter__(self) -> tp.Iterator[str]:
        if not self.__built__:
            raise RuntimeError('Registry is not build yet.')
        return iter(self._registry)

    def __len__(self) -> int:
        if not self.__built__:
            raise RuntimeError('Registry is not build yet.')
        return len(self._registry)

    def items(self) -> ItemsView[str, RegistryEntry]:
        return ItemsView(self)

    def register(self, name: str, cls: type[object], path: list[str] | None = None):
        """
            Register new registry_base_type.
        """
        if self.__built__:
            self._register(name, cls, path)
        else:
            # Delay registration until all default objects were identified.
            if name in self._raw_registry:
                raise NameError(f'{self._registry_base_type} name "{name}" is already queued to be register.')
            self._raw_registry[name] = cls

    def _register(self, name: str, cls: type[object], path: list[str] | None = None):
        """
            Validate and register new item.
        """
        if self._registry_base_type == validation.DEFAULT_INITIALIZER_PATH:
            if not validation._is_initializer(cls):
                raise TypeError(f'Tried to register "{cls.__name__}" under the label "{name}", but '
                                f'"{cls.__name__}" is not a valid Initializer.')
        else:
            if not validation._is_spark_type(cls, self._registry_base_type):
                raise TypeError(f'Tried to register "{cls.__name__}" under the label "{name}", but '
                                f'"{cls.__name__}" does not inherit from {self._registry_base_type}.')
        if self._exists(name):
            raise ValueError(f'Tried to register "{cls.__name__}" under the label "{name}", but '
                             f'name "{name}" is already registered to another class.')
        if not path is None:
            if not isinstance(path, list):
                raise TypeError(f'Expect path to be a list of str but got {type(path).__name__}.')
            for p in path:
                if not isinstance(p, str):
                    raise TypeError(f'Expect path to be a list of str but found item of type {type(p).__name__}.')
        # Register
        name = normalize_name(name)
        path = self._get_default_path(cls) if path is None else path
        self._leaf_class.add(cls.__name__)
        self._registry[name] = RegistryEntry(name=name, class_ref=cls, path=path)
        logging.info(f'Registered "{name}" to class "{cls.__name__}" with path "{path}".')

    def _build(self):
        """
            Build registry.
        """
        if self.__built__:
            return
        # NOTE: This code is only be accessible to internal classes. 
        # User definitions are routed to the register method.
        for name, cls in self._raw_registry.items():
            self._register(name, cls)
        self.__built__ = True
        del self._raw_registry
        logging.info(f'Register built successfully.')

    def get(self, name: str) -> RegistryEntry | None:
        """
            Safely retrieves a component entry by name.
        """
        if self.__built__:
            return self._registry.get(normalize_name(name), None)
        else: 
            raise RuntimeError(f'Registry is not yet built. Registry must be built first before trying to access it.')

    def get_by_cls(self, cls: type) -> RegistryEntry | None:
        """
            Safely retrieves a component entry by name.
        """
        if self.__built__:
            for value in self._registry.values():
                if value.class_ref == cls:
                    return value
            return None
        else: 
            raise RuntimeError(f'Registry is not yet built. Registry must be built first before trying to access it.')

    def _exists(self, name):
        if name in self._registry:
            return True
        return False

    def _get_default_path(self, cls: tp.Any):
        if self._registry_base_type == validation.DEFAULT_INITIALIZER_PATH:
            name = cls.__module__.split('.')[-1]
            name_map = INITIALIZERS_ALIAS_MAP.get(name, name)
            path = ['Initializers', name_map]
            return path
        else:
            path = []
            for base in cls.__mro__:
                # Start from the class
                if base in [cls]:
                    continue
                # Skip inheritance chains of classes that are leaves
                if base.__name__ in self._leaf_class:
                    continue
                # Stop at registry_base_type
                if base.__name__ in [self._registry_base_type_name]:
                    break
                name = base.__name__
                name_map = MRO_PATH_ALIAS_MAP.get(name, name)
                if name_map:
                    path.append(name_map)
            return path[::-1]

    @property
    def is_finalized(self) -> bool:
        return self.__built__

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Registry():

    def __init__(self):
        self.MODULES = SubRegistry(registry_base_type=validation.DEFAULT_SPARKMODULE_PATH)
        self.PAYLOADS = SubRegistry(registry_base_type=validation.DEFAULT_PAYLOAD_PATH)
        self.INITIALIZERS = SubRegistry(registry_base_type=validation.DEFAULT_INITIALIZER_PATH)
        self.CONFIG = SubRegistry(registry_base_type=validation.DEFAULT_INITIALIZER_PATH)
        self.CFG_VALIDATORS = SubRegistry(registry_base_type=validation.DEFAULT_CFG_VALIDATOR_PATH)

    def _build(self,):
        self.MODULES._build()
        self.PAYLOADS._build()
        self.INITIALIZERS._build()
        self.CONFIG._build()
        self.CFG_VALIDATORS._build()

# Default Instance
REGISTRY = Registry()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Decorator method for modules.
def register_module(arg: SparkModule | str | None = None):
    def decorator(cls):
        name = arg if isinstance(arg, str) else cls.__name__
        REGISTRY.MODULES.register(cls=cls, name=name)
        return cls
    if callable(arg):
        # Decorator was used as "", arg is the class itself. 
        return decorator(arg)
    else:
        # Decorator was used as "('somename')" or "()", in this case, 'arg' is the name (or None).
        return decorator

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Decorator method for payloads.
def register_payload(arg: SparkPayload | str | None = None):
    def decorator(cls):
        name = arg if isinstance(arg, str) else cls.__name__
        REGISTRY.PAYLOADS.register(cls=cls, name=name)
        return cls
    if callable(arg):
        # Decorator was used as "", arg is the class itself. 
        return decorator(arg)
    else:
        # Decorator was used as "('somename')" or "()", in this case, 'arg' is the name (or None).
        return decorator

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Decorator method for payloads.
def register_initializer(arg: Initializer | str | None = None):
    def decorator(cls):
        name = arg if isinstance(arg, str) else cls.__name__
        REGISTRY.INITIALIZERS.register(cls=cls, name=name)
        return cls
    if callable(arg):
        # Decorator was used as "", arg is the class itself. 
        return decorator(arg)
    else:
        # Decorator was used as "('somename')" or "()", in this case, 'arg' is the name (or None).
        return decorator

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Decorator method for configuration validators.
def register_config(arg: BaseSparkConfig | str | None = None):
    def decorator(cls):
        name = arg if isinstance(arg, str) else cls.__name__
        REGISTRY.CONFIG.register(cls=cls, name=name)
        return cls
    if callable(arg):
        # Decorator was used as "", arg is the class itself. 
        return decorator(arg)
    else:
        # Decorator was used as "('somename')" or "()", in this case, 'arg' is the name (or None).
        return decorator

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Decorator method for configuration validators.
def register_cfg_validator(arg: ConfigurationValidator | str | None = None):
    def decorator(cls):
        name = arg if isinstance(arg, str) else cls.__name__
        REGISTRY.CFG_VALIDATORS.register(cls=cls, name=name)
        return cls
    if callable(arg):
        # Decorator was used as "", arg is the class itself. 
        return decorator(arg)
    else:
        # Decorator was used as "('somename')" or "()", in this case, 'arg' is the name (or None).
        return decorator
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

# NOTE: This aliases are used by the Graph Editor to create pretty context menus.

MRO_PATH_ALIAS_MAP = {
    # Aliases
    'Interface': 'Interfaces',
    'InputInterface': 'Input',
    'OutputInterface': 'Output',
    'ControlFlowInterface': 'Control',
    'Component': 'Components',
    'Delays': 'Delays',
    #'Plasticity': 'Learning Rules',
    'Soma': 'Somas',
    'Synanpses': 'Synanpses',
    'Neuron': 'Neurons',
    # Exclusions
    'ValueSparkPayload': None,
}

INITIALIZERS_ALIAS_MAP = {
    # Aliases
    'kernel': 'Kernel',
    'delay': 'Dealy',
}

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################