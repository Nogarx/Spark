#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp
if tp.TYPE_CHECKING:
    from spark.nn.controllers.neuron import Neuron, NeuronConfig

import pathlib as pl
import logging
import dataclasses as dc
import typing as tp
import copy
from collections.abc import Mapping, ItemsView
import spark.core.utils as utils
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

    def __init__(self, registry_base_type: str) -> None:
        self._raw_registry: dict[str, type[object]] = {}
        self._registry: dict[str, RegistryEntry] = {}
        self._leaf_class = set()
        self.__built__ = False
        self._registry_base_type = registry_base_type
        self._registry_base_type_name = registry_base_type.split('.')[-1]

    def __getitem__(self, key: str) -> RegistryEntry:
        if not self.__built__:
            raise RuntimeError('Registry is not build yet.')
        return copy.deepcopy(self._registry[key])

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
                raise NameError(
                    f'{self._registry_base_type} name \"{name}\" is already queued to be register.'
                )
            self._raw_registry[name] = cls

    def _register(self, name: str, cls: type[object], path: list[str] | None = None):
        """
            Validate and register new item.
        """
        # Special case for initializers
        if self._registry_base_type == validation.DEFAULT_INITIALIZER_PATH:
            if not validation._is_initializer_type(cls):
                raise TypeError(f'Tried to register "{cls.__name__}" under the label "{name}", but '
                                f'"{cls.__name__}" is not a valid Initializer.')
        # Special case for modules + controllers
        elif self._registry_base_type == validation.DEFAULT_SPARK_MODULE_PATH:
            if not (
                validation._is_spark_type(cls, self._registry_base_type) or 
                validation._is_spark_type(cls, validation.DEFAULT_SPARK_CONTROLLER_PATH)
                ):
                raise TypeError(f'Tried to register "{cls.__name__}" under the label "{name}", but '
                                f'"{cls.__name__}" does not inherit from {self._registry_base_type}.')
        # Everything else
        else:
            pass
            #if not validation._is_spark_type(cls, self._registry_base_type):
            #    raise TypeError(f'Tried to register "{cls.__name__}" under the label "{name}", but '
            #                    f'"{cls.__name__}" does not inherit from {self._registry_base_type}.')
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
        name = utils.normalize_str(name)
        path = self._get_default_path(cls) if path is None else path
        self._leaf_class.add(cls.__name__)
        self._registry[name] = RegistryEntry(name=name, class_ref=cls, path=path)
        logging.info(f'Registered "{name}" to class "{cls.__name__}" with path "{path}".')

    def _build(self) -> None:
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

    def get(self, name: str, default: tp.Any = None) -> RegistryEntry | None:
        """
            Safely retrieves a component entry by name.
        """
        if self.__built__:
            return self._registry.get(utils.normalize_str(name), default)
        else: 
            raise RuntimeError(f'Registry is not yet built. Registry must be built first before trying to access it.')

    def get_by_cls(self, cls: type | None) -> RegistryEntry | None:
        """
            Safely retrieves a component entry by name.
        """
        # Accepting None's avoids a lot of extra type checks.
        if isinstance(cls, type(None)):
            return None
        if self.__built__:
            for value in self._registry.values():
                if value.class_ref == cls:
                    return value
            return None
        else: 
            raise RuntimeError(f'Registry is not yet built. Registry must be built first before trying to access it.')

    def _exists(self, name) -> bool:
        if name in self._registry:
            return True
        return False

    def exists(self, name) -> bool:
        if utils.normalize_str(name) in self._registry:
            return True
        return False

    def _get_default_path(self, cls: tp.Any):
        if self._registry_base_type == validation.DEFAULT_INITIALIZER_PATH:
            name = cls.__module__.split('.')[-1]
            name_map = INITIALIZERS_ALIAS_MAP.get(name, name)
            path = ['Initializers', name_map]
            return path
        elif self._registry_base_type == validation.DEFAULT_SPARK_NEURON_PATH:
            name = cls.__module__.split('.')[-1]
            path = ['Neurons']
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
    """
        Registry object.
    """
    
    def __init__(self):
        self.MODULES = SubRegistry(registry_base_type=validation.DEFAULT_SPARK_MODULE_PATH)
        self.NEURONS = SubRegistry(registry_base_type=validation.DEFAULT_SPARK_NEURON_PATH)
        self.PAYLOADS = SubRegistry(registry_base_type=validation.DEFAULT_PAYLOAD_PATH)
        self.INITIALIZERS = SubRegistry(registry_base_type=validation.DEFAULT_INITIALIZER_PATH)
        self.CONFIG = SubRegistry(registry_base_type=validation.DEFAULT_CONFIG_PATH)
        self.CFG_VALIDATORS = SubRegistry(registry_base_type=validation.DEFAULT_CFG_VALIDATOR_PATH)

    def _build(self,):
        self.MODULES._build()
        self.NEURONS._build()
        self.PAYLOADS._build()
        self.INITIALIZERS._build()
        self.CONFIG._build()
        self.CFG_VALIDATORS._build()

# Default Instance
REGISTRY = Registry()
"""
    Registry singleton.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def create_registry_decorator(
        sub_registry: SubRegistry,
        base_class_name: str,
        base_class_path: str,
        base_class_abr: str | None = None
    ):

    T = tp.TypeVar("T")

    @tp.overload
    def register(arg: str | None) -> tp.Callable[[type[T]], type[T]]:
        pass

    @tp.overload
    def register(arg: type[T]) -> type[T]:
        pass

    def register(arg: type[T] | str | None = None) -> tp.Callable[[type[T]], type[T]] | type[T]:
        def decorator(cls: type[T]) -> type[T]:
            name = arg if isinstance(arg, str) else cls.__name__
            sub_registry.register(cls=cls, name=name)
            return cls
        if callable(arg):
            # Called as @register_module, arg is the class itself
            return decorator(arg)
        else:
            # Called as @register_module('name') or @register_module, arg is str or None
            return decorator
        
    abbreviation = f'{base_class_abr} ({base_class_path})' if base_class_abr else base_class_path
    docstring = f"""
        Decorator used to register a new {base_class_name}. 
        Note that module must inherit from {abbreviation}
    """

    register.__doc__ = docstring
    return register

#-----------------------------------------------------------------------------------------------------------------------------------------------#


register_module = create_registry_decorator(
    sub_registry=REGISTRY.MODULES, 
    base_class_name='SparkModule', 
    base_class_path='spark.core.module.SparkModule',
    base_class_abr='spark.nn.Module'
)
"""
    Decorator used to register a new SparkModule. 
    Note that module must inherit from spark.nn.Module (spark.core.module.SparkModule)
"""

register_neuron = create_registry_decorator(
    sub_registry=REGISTRY.NEURONS, 
    base_class_name='Neuron', 
    base_class_path='spark.nn.controllers.neuron.Neuron',
    base_class_abr='spark.nn.Neuron'
)
"""
    Decorator used to register a new Neuron model. 
    Note that module must inherit from spark.nn.Neuron (spark.nn.controllers.neuron.Neuron)
"""

register_payload = create_registry_decorator(
    sub_registry=REGISTRY.PAYLOADS, 
    base_class_name='SparkPayload', 
    base_class_path='spark.core.payloads.SparkPayload',
    base_class_abr='spark.SparkPayload'
)
"""
    Decorator used to register a new SparkPayload. 
    Note that module must inherit from spark.SparkPayload (spark.core.payloads.SparkPayload)
"""

register_initializer = create_registry_decorator(
    sub_registry=REGISTRY.INITIALIZERS, 
    base_class_name='Initializer', 
    base_class_path='spark.nn.initializers.base.Initializer',
)
"""
    Decorator used to register a new Initializer. 
    Note that module must inherit from spark.nn.initializers.base.Initializer
"""

register_config = create_registry_decorator(
    sub_registry=REGISTRY.CONFIG, 
    base_class_name='SparkConfig', 
    base_class_path='spark.core.config.SparkConfig',
    base_class_abr='spark.nn.BaseConfig'
)
"""
    Decorator used to register a new SparkConfig. 
    Note that module must inherit from spark.nn.BaseConfig (spark.core.config.SparkConfig)
"""

register_cfg_validator = create_registry_decorator(
    sub_registry=REGISTRY.CFG_VALIDATORS, 
    base_class_name='ConfigurationValidator', 
    base_class_path='spark.core.config_validation.ConfigurationValidator',
)
"""
    Decorator used to register a new ConfigurationValidator. 
    Note that module must inherit from spark.core.config_validation.ConfigurationValidator
"""

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
    'Plasticity': 'Plasticity Rules',
    'Soma': 'Somas',
    'Synapses': 'Synapses',
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

def _construct_neuron_config_cls(cls_name: str, config: NeuronConfig) -> type[NeuronConfig]:
    """
        Generate a NeuronConfig subclass programmatically from a NeuronConfig instance.
    """
    from spark.nn.controllers.neuron import NeuronConfig
    # Shallow copy
    config = copy.deepcopy(config)
    # Cls namespace
    cls_name = f'{cls_name}Config'
    ns_annotations: dict[str, tp.Any] = {}
    namespace: dict[str, tp.Any] = {}
    # Grab config fields
    for field in dc.fields(config):
        namespace[field.name] = getattr(config, field.name, None)
        ns_annotations[field.name] = field.type
    # Copy metadata
    namespace['__metadata__'] = getattr(config, '__metadata__', {})
    namespace['__graph_editor_metadata__'] = getattr(config, '__graph_editor_metadata__', {})
    namespace['__annotations__'] = ns_annotations
    # Create class and link it to spark
    neuron_config_cls = type(cls_name, (NeuronConfig,), namespace)
    neuron_config_cls.__module__ = 'spark'
    return neuron_config_cls

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _construct_neuron_cls(cls_name: str, config_cls: type[NeuronConfig]) -> type[Neuron]:
    """
        Generate a Neuron subclass programmatically from a NeuronConfig type.
    """
    from spark.nn.controllers.neuron import Neuron
    # Cls namespace
    cls_name = f'{cls_name}'
    ns_annotations: dict[str, tp.Any] = {'config': config_cls}
    namespace: dict[str, tp.Any] = {}
    namespace['__annotations__'] = ns_annotations
    # Create class and link it to spark
    neuron_cls = type(cls_name, (Neuron,), namespace)
    neuron_cls.__module__ = 'spark'
    return neuron_cls

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: Clean up is necessary in case something fails in order to prevent orphaned pairs.
def register_neuron_from_config(cls_name: str, config: NeuronConfig) -> None:
    """
        Generate a (Neuron, NeuronConfig) subclass pair programmatically from a NeuronConfig instance.
    """
    from spark.nn.controllers.neuron import NeuronConfig
    if REGISTRY.NEURONS.exists(cls_name):
        raise KeyError(
            f'Unable to generate a (Neuron, NeuronConfig) subclass pair. The name {cls_name} is already in use by another class in the registry.'
        )

    if not isinstance(config, NeuronConfig):
        raise TypeError(
            f'Expected "config" to be of type "{NeuronConfig.__name__}" but got type "{type(config).__name__}".'
        )
    try:
        config_cls = _construct_neuron_config_cls(cls_name, config)
        register_config(config_cls)
    except Exception as e:
        raise RuntimeError(
            f'Unable to generate a configuration class from "config". Error: {e}.'
        )
    
    try:
        neuron_cls = _construct_neuron_cls(cls_name, config_cls)
        register_neuron(neuron_cls)
    except Exception as e:
        raise RuntimeError(
            f'Unable to generate a configuration class from "config". Error: {e}.'
        )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def register_neuron_from_config_file(cls_name: str, path: pl.Path) -> None:
    """
        Generate a (Neuron, NeuronConfig) subclass pair programmatically from a NeuronConfig file.
    """
    from spark.nn.controllers.neuron import NeuronConfig
    path = pl.Path(path).absolute()
    if path.exists():
        try:
            config_instance = NeuronConfig.from_file(path)
        except:
            raise RuntimeError(
                f'Unable to read "{path}" as a NeuronConfig object.'
            )
        register_neuron_from_config(cls_name, config_instance)
    else:
        raise RuntimeError(
            f'Invalid path: "{path}".'
        )
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################