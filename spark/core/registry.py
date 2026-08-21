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
import enum
import importlib
from collections.abc import Mapping, ItemsView
from types import MappingProxyType
import spark.core.utils as utils
import spark.core.validation as validation

logger = logging.getLogger('spark')

# TODO: Reintroduce register validations.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class RegistryNamespace(enum.Enum):
    Components = enum.auto()
    Initializers = enum.auto()
    Payloads = enum.auto()
    Interfaces = enum.auto()
    Neurons = enum.auto()
    Configs = enum.auto()
    Validators = enum.auto()

    @classmethod
    def base_module(cls, namespace: RegistryNamespace) -> str:
        if namespace == RegistryNamespace.Components:
            return 'spark.core.module.SparkModule'
        elif namespace == RegistryNamespace.Initializers:
            return 'spark.nn.initializers.base.Initializer'
        elif namespace == RegistryNamespace.Payloads:
            return 'spark.core.payloads.SparkPayload'
        elif namespace == RegistryNamespace.Interfaces:
            return 'spark.nn.interfaces.base.Interface'
        elif namespace == RegistryNamespace.Neurons:
            return 'spark.nn.controllers.neuron.Neuron'
        elif namespace == RegistryNamespace.Configs:
            return 'spark.core.config.SparkConfig'
        elif namespace == RegistryNamespace.Validators:
            return 'spark.core.config_validation.ConfigurationValidator'
        else:
            raise RuntimeError(
                f'"{namespace}" is not a valid RegistryNamespace.'
            )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass(frozen=True)
class RegistryEntry:
    """
        Structured entry for the registry.
    """
    name: str
    module: str
    qualname: str
    namespace: RegistryNamespace
    path: list[str]
    metadata: dict[str, tp.Any]

    def get_cls(self,) -> type:
        module = importlib.import_module(self.module)
        return getattr(module, self.qualname)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SubRegistry:
    """
        One namespace of a registry, as its own mapping of name to entry.
    """

    def __init__(self, instance: Registry,  namespace: RegistryNamespace) -> None:
        self._instance = instance
        self._namespace = namespace

    def _entries(self) -> dict[str, RegistryEntry]:
        """
            Entries of this namespace, keyed by their normalized name.
        """
        return self._instance._registry[self._namespace]

    def get(self, key: str | None, default: tp.Any = None) -> RegistryEntry | None:
        """
            Entry registered under a name.

            Args:
                key: str | None, name to look up.
                default: tp.Any, what to answer with when the name is not registered.

            Returns:
                RegistryEntry | None, the entry, or the default.
        """
        self._instance._require_built()
        if not isinstance(key, str) or not key:
            return default
        return self._entries().get(utils.normalize_str(key), default)

    def get_by_cls(self, cls: type, default: tp.Any = None) -> RegistryEntry | None:
        """
            Entry a class is registered under.

            Args:
                cls: type, the registered class.
                default: tp.Any, what to answer with when the class is not registered.

            Returns:
                RegistryEntry | None, the entry, or the default.
        """
        self._instance._require_built()
        if not isinstance(cls, type):
            return default
        for entry in self._entries().values():
            if entry.module == cls.__module__ and entry.qualname == cls.__qualname__:
                return entry
        return self.get(cls.__name__, default)

    def __getitem__(self, key: str) -> RegistryEntry:
        return self._entries()[utils.normalize_str(key)]

    def __setitem__(self, key: str, value: RegistryEntry) -> None:
        self._entries()[utils.normalize_str(key)] = value

    def __contains__(self, key: str) -> bool:
        return isinstance(key, str) and bool(key) and utils.normalize_str(key) in self._entries()

    def __iter__(self) -> tp.Iterator[str]:
        return iter(self._entries())

    def __len__(self) -> int:
        return len(self._entries())

    def values(self) -> tp.ValuesView[RegistryEntry]:
        return self._entries().values()
    
    def keys(self) -> tp.KeysView[str]:
        return self._entries().keys()
    
    def items(self) -> ItemsView[str, RegistryEntry]:
        return self._entries().items()

    def register(self, name: str, cls: type[object], path: list[str] | None = None) -> None:
        self._instance.register(self._namespace, name, cls, path)

    def exists(self, name: str) -> bool:
        return self._instance.exists(self._namespace, name)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Registry:
    """
        Generic registry implementation.
    """

    if tp.TYPE_CHECKING:
        _raw_registry: utils.TwoKeyDict[RegistryNamespace, str, type[object]]
        _registry: utils.TwoKeyDict[RegistryNamespace, str, RegistryEntry]

    def __init__(self,) -> None:
        self._raw_registry = utils.TwoKeyDict({r: {} for r in RegistryNamespace._member_map_.values()})
        self._registry = utils.TwoKeyDict({r: {} for r in RegistryNamespace._member_map_.values()})
        self.__built__ = False

    def __getattr__(self, name: str) -> SubRegistry:
        """
            Serves "REGISTRY.<Namespace>" as a view of that namespace.
        """
        try:
            namespace = RegistryNamespace[name]
        except KeyError:
            raise AttributeError(f'"{type(self).__name__}" has no attribute "{name}".') from None
        attribute = f'_{namespace.name}'
        subregistry = self.__dict__.get(attribute, None)
        if subregistry is None:
            subregistry = SubRegistry(self, namespace)
            setattr(self, attribute, subregistry)
        return subregistry

    def _require_built(self) -> None:
        """
            Raises unless the registry is built.
        """
        if not self.__built__:
            raise RuntimeError(
                f'Registry is not yet built. Registry must be built first before trying to access it.'
            )

    def __iter__(self) -> tp.Iterator[tuple[RegistryNamespace, SubRegistry]]:
        if not self.__built__:
            raise RuntimeError('Registry is not build yet.')
        def iterator() -> tp.Generator[tuple[RegistryNamespace, SubRegistry], tp.Any, None]:
            for namespace in RegistryNamespace._member_map_.values():
                yield namespace, getattr(self, namespace.name)
        return iterator()
    
    def __len__(self) -> int:
        if not self.__built__:
            raise RuntimeError('Registry is not build yet.')
        return len(self._registry)

    def entries(self) -> ItemsView[tuple[RegistryNamespace, str], SubRegistry]:
        if not self.__built__:
            raise RuntimeError('Registry is not build yet.')
        return self._registry.items()

    def register(self, namespace: RegistryNamespace, name: str, cls: type[object], path: list[str] | None = None) -> None:
        """
            Register new registry_base_type.
        """
        if self.__built__:
            self._register(namespace, name, cls, path)
        else:
            # Delay registration until all default objects were identified.
            if (namespace, name) in self._raw_registry:
                raise NameError(
                    f'{namespace.name} \"{name}\" is already queued to be register.'
                )
            self._raw_registry[namespace, name] = cls

    def _register(self, namespace: RegistryNamespace | str, name: str, cls: type[object], path: list[str] | None = None, metadata: dict[str, tp.Any] | None = None) -> None:
        """
            Validate and register new item.
        """
        if isinstance(namespace, str):
            namespace = RegistryNamespace[namespace]
        name = utils.normalize_str(name)
        if self.exists(namespace, name):
            raise ValueError(f'Tried to register "{cls.__name__}" under the label "{name}", but '
                            f'name "{name}" is already registered to another class.')
        if not path is None:
            if isinstance(path, tuple):
                path = list(path)
            if not isinstance(path, list):
                raise TypeError(f'Expect path to be a list of str but got {type(path).__name__}.')
            for p in path:
                if not isinstance(p, str):
                    raise TypeError(f'Expect path to be a list of str but found item of type {type(p).__name__}.')
        # Register
        path = self._get_default_path(namespace, cls) if path is None else path
        self._registry[namespace, name] = RegistryEntry(
            name=name, 
            module=cls.__module__, 
            qualname=cls.__qualname__,
            namespace=namespace,
            path=path,
            metadata={} if metadata is None else metadata,
        )
        logger.info(f'Registered "{name}" to class "{cls.__name__}" with path "{path}".')

    def _build(self) -> None:
        """
            Build registry.
        """
        if self.__built__:
            return
        # NOTE: This code is only be accessible to internal classes. 
        # User definitions are routed to the register method.
        for (namespace, name), cls in self._raw_registry.items():
            self._register(namespace, name, cls)
        self.__built__ = True
        del self._raw_registry
        logger.info(f'Register built successfully.')

    def get(self, namespace: RegistryNamespace | str, name: str, default: tp.Any = None) -> RegistryEntry | None:
        """
            Safely retrieves a component entry by name.
        """
        self._require_built()
        if isinstance(namespace, str):
            namespace = RegistryNamespace[namespace]
        if not isinstance(name, str) or not name:
            return default
        return self._registry.get((namespace, utils.normalize_str(name)), default)
        
    def exists(self, namespace: RegistryNamespace | str, name: str) -> bool:
        if isinstance(namespace, str):
            namespace = RegistryNamespace[namespace]
        if not isinstance(name, str) or not name:
            return False
        if (namespace, utils.normalize_str(name)) in self._registry:
            return True
        return False

    def _get_default_path(self, namespace: RegistryNamespace, cls: tp.Any) -> tuple[str, ...]:
        if namespace == RegistryNamespace.Initializers:
            return ['Initializers']
        elif namespace == RegistryNamespace.Neurons:
            return ['Neurons']
        else:
            path = []
            for base in cls.__mro__:
                # Start from the class
                if base in [cls]:
                    continue
                # Stop at registry_base_type
                base_type_name = RegistryNamespace.base_module(namespace).split('.')[-1]
                if base.__name__ == base_type_name:
                    break
                name = base.__name__
                name_map = MRO_PATH_ALIAS_MAP.get(name, name)
                # Check if is a simple name map or a tuple name map
                if isinstance(name_map, str):
                    path.append(name_map)
                elif isinstance(name_map, tuple):
                    path += list(name_map)
            return path[::-1]

    @property
    def is_built(self) -> bool:
        return self.__built__

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkRegistry(Registry):
    """
        Generic registry implementation.
    """

    if tp.TYPE_CHECKING:
        Components: SubRegistry
        Initializers: SubRegistry
        Payloads: SubRegistry
        Interfaces: SubRegistry
        Neurons: SubRegistry
        Configs: SubRegistry
        Validators: SubRegistry

    def __init__(self,) -> None:
        super().__init__()
        self._raw_registry = utils.TwoKeyDict({r: {} for r in RegistryNamespace._member_map_.values()})
        self._registry = utils.TwoKeyDict({r: {} for r in RegistryNamespace._member_map_.values()})

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Default Instance
REGISTRY = SparkRegistry()
"""
    Registry singleton.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def create_registry_decorator(
        namespace: RegistryNamespace,
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
            subregistry: SubRegistry = getattr(REGISTRY, namespace.name)
            subregistry.register(cls=cls, name=name)
            return cls
        if callable(arg):
            # Called as @register_module, arg is the class itself
            return decorator(arg)
        else:
            # Called as @register_module('name') or @register_module, arg is str or None
            return decorator

    base_module = RegistryNamespace.base_module(namespace)
    docstring = f"""
        Decorator used to register a new {base_module}. 
    """

    register.__doc__ = docstring
    return register

#-----------------------------------------------------------------------------------------------------------------------------------------------#


register_module = create_registry_decorator(
    namespace=RegistryNamespace.Components, 
)
"""
    Decorator used to register a new SparkModule. 
    Note that module must inherit from spark.nn.Module (spark.core.module.SparkModule)
"""

register_neuron = create_registry_decorator(
    namespace=RegistryNamespace.Neurons, 
)
"""
    Decorator used to register a new Neuron model. 
    Note that module must inherit from spark.nn.Neuron (spark.nn.controllers.neuron.Neuron)
"""

register_payload = create_registry_decorator(
    namespace=RegistryNamespace.Payloads, 
)
"""
    Decorator used to register a new SparkPayload. 
    Note that module must inherit from spark.SparkPayload (spark.core.payloads.SparkPayload)
"""

register_interface = create_registry_decorator(
    namespace=RegistryNamespace.Interfaces, 
)
"""
    Decorator used to register a new Interface. 
    Note that module must inherit from spark.nn.interfaces.base.Interface
"""

register_initializer = create_registry_decorator(
    namespace=RegistryNamespace.Initializers, 
)
"""
    Decorator used to register a new Initializer. 
    Note that module must inherit from spark.nn.initializers.base.Initializer
"""

register_config = create_registry_decorator(
    namespace=RegistryNamespace.Configs, 
)
"""
    Decorator used to register a new SparkConfig. 
    Note that module must inherit from spark.nn.BaseConfig (spark.core.config.SparkConfig)
"""

register_cfg_validator = create_registry_decorator(
    namespace=RegistryNamespace.Validators, 
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
    'InputInterface': ('Input', 'Interfaces'),
    'OutputInterface': ('Output', 'Interfaces'),
    'ControlInterface': ('Control', 'Interfaces'),
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

def _bind_to_spark(cls: type) -> type:
    """
        Publishes a class built at runtime under the "spark" namespace.

        Args:
            cls: type, the class to publish.

        Returns:
            type, the same class.
    """
    import spark as spark_module
    name = cls.__name__
    existing = getattr(spark_module, name, None)
    if existing is not None and existing is not cls:
        raise NameError(
            f'Unable to publish "{name}" under "spark": the name is already taken by another object.'
        )
    cls.__module__ = spark_module.__name__
    setattr(spark_module, name, cls)
    return cls

#-----------------------------------------------------------------------------------------------------------------------------------------------#

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
    return _bind_to_spark(neuron_config_cls)

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
    return _bind_to_spark(neuron_cls)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# TODO: Clean up is necessary in case something fails in order to prevent orphaned pairs.
def register_neuron_from_config(cls_name: str, config: NeuronConfig) -> None:
    """
        Generate a (Neuron, NeuronConfig) subclass pair programmatically from a NeuronConfig instance.
    """
    from spark.nn.controllers.neuron import NeuronConfig
    if REGISTRY.Neurons.exists(cls_name):
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

def register_models_from_payload(payload: tp.Any) -> list[str]:
    """
        Registers a collection of models from decoded json documents.

        Args:
            payload: tp.Any, a decoded json document, before the spark decoder has read it.

        Returns:
            list[str], names of the models that were registered.
    """
    import json
    from spark.core.serializer import SparkJSONDecoder
    from spark.nn.controllers.neuron import NeuronConfig

    definable = {'Neurons': NeuronConfig}
    missing: dict[str, tuple[str, tp.Any]] = {}

    def collect(node: tp.Any) -> None:
        if isinstance(node, dict):
            data = node.get('__data__') if node.get('__type__') == 'module_specs' else None
            if isinstance(data, dict):
                reference = data.get('module_cls') or {}
                name = reference.get('__module_type__')
                namespace = reference.get('__subregistry__')
                subregistry = getattr(REGISTRY, namespace, None) if namespace else None
                if name and namespace in definable and subregistry and not subregistry.get(name):
                    if data.get('config') is not None:
                        missing[name] = (namespace, data['config'])
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(payload)
    registered = []
    for name, (namespace, config_payload) in missing.items():
        base_cls = definable[namespace]
        base_entry = REGISTRY.Configs.get_by_cls(base_cls)
        if not base_entry:
            continue
        generic = dict(config_payload)
        generic['__type__'] = base_entry.name
        try:
            config = json.loads(json.dumps(generic), cls=SparkJSONDecoder)
            register_neuron_from_config(name, config)
        except Exception as error:
            logger.warning(f'Unable to build the model "{name}" from the definition in the file. Error: {error}.')
            continue
        registered.append(name)
    return registered

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