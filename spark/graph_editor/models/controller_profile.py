#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.core.config import SparkConfig
    from spark.core.specs import PortSpecs
    from spark.nn.controllers.base import Controller

import logging
import typing as tp
import dataclasses as dc
import spark.core.utils as utils
from spark.core.registry import REGISTRY, RegistryNamespace, RegistryEntry
from spark.graph_editor.styles import resources as icons

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# NOTE: A graph in the editor is always the template of one Controller. Everything that depends on which
# controller is being built is declared here, so that the rest of the editor never branches on a controller
# type. Most of the behaviour is derived from the controller class itself (its configuration class, its
# properties, its recurrence rules); only what cannot be introspected is declared:
#   - palette_namespaces: which registry namespaces may be placed on the canvas.
#   - atomic_namespaces:  which of those are served as a single node instead of being expanded into the
#                         components they are made of. This is what makes an imported neuron a single node in
#                         a Brain and a collection of somas/synapses/delays in a Neuron.
#   - import_namespaces:  which registered models can be expanded into this graph. They are not placed as a
#                         node: their modules become nodes of the current graph.
#   - model_namespace:    where models built under this profile are registered, so another profile can tell
#                         whether it is able to host them. A Brain is not hosted by anything, so it has none.
# Supporting a new controller is one more registration below.

@dc.dataclass(frozen=True)
class ControllerProfile:
    """
        Describes how the editor behaves while building a given controller.
    """

    key: str
    label: str
    icon: str  # resource path, see spark.graph_editor.styles.resources
    summary: str
    controller_name: str
    palette_namespaces: tuple[RegistryNamespace, ...]
    atomic_namespaces: tuple[RegistryNamespace, ...]
    import_namespaces: tuple[RegistryNamespace, ...] = ()
    model_namespace: RegistryNamespace | None = None
    cache_based: bool = False

    #-------------------------------------------------------------------------------------------------------#

    @property
    def registry_entry(self) -> RegistryEntry | None:
        """
            Registry entry of the controller backing this profile.
        """
        try:
            return REGISTRY.Components.get(utils.normalize_str(self.controller_name))
        except KeyError:
            logger.error(f'Controller "{self.controller_name}" is not registered.')
            return None

    @property
    def controller_cls(self) -> type[Controller] | None:
        """
            Controller class backing this profile.
        """
        entry = self.registry_entry
        return entry.get_cls() if entry is not None else None

    @property
    def config_cls(self) -> type[SparkConfig] | None:
        """
            Configuration class produced by a graph built under this profile.
        """
        controller_cls = self.controller_cls
        if controller_cls is None:
            return None
        return controller_cls.get_config_spec()

    def self_property_specs(self) -> dict[str, PortSpecs]:
        """
            Properties the controller exposes to its own modules (the "__self__" origin of a PortMap).
        """
        controller_cls = self.controller_cls
        if controller_cls is None:
            return {}
        try:
            return controller_cls._get_controller_property_specs()
        except Exception as error:
            logger.warning(f'Unable to resolve the properties of "{self.label}": {error}')
            return {}

    def is_importable(self, namespace: RegistryNamespace) -> bool:
        """
            True if members of the namespace are expanded into their own modules instead of being placed.
        """
        return namespace in self.import_namespaces

    def is_atomic(self, namespace: RegistryNamespace) -> bool:
        """
            True if members of the namespace are placed as a single node instead of being expanded.
        """
        return namespace in self.atomic_namespaces

    def hosts(self, other: 'ControllerProfile') -> bool:
        """
            True if a model built under "other" belongs on this canvas as a single node.

            A Brain hosts Neurons this way. It is not the same as importing: an imported model is
            expanded into the modules it is made of, a hosted one stays whole.
        """
        namespace = other.model_namespace
        if namespace is None:
            return False
        return namespace in self.palette_namespaces and self.is_atomic(namespace)

    def allows_cycle(self, module_cls: type | None = None) -> bool:
        """
            True if a self/backwards connection is legal under this profile.

            Cache based controllers read every input from the previous timestep, so any cycle is legal.
            Otherwise the target module must define a recurrent contract.
        """
        if self.cache_based:
            return True
        if module_cls is None:
            return False
        try:
            return bool(module_cls.has_recurrent_contract())
        except Exception:
            return False

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

CONTROLLER_PROFILES: dict[str, ControllerProfile] = {}
"""
    Controller profiles available to the editor, keyed by profile key.
"""

def register_controller_profile(profile: ControllerProfile) -> ControllerProfile:
    """
        Registers a new controller profile.
    """
    if profile.key in CONTROLLER_PROFILES:
        raise KeyError(f'A controller profile is already registered under the key "{profile.key}".')
    CONTROLLER_PROFILES[profile.key] = profile
    return profile

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_controller_profile(key: str | None) -> ControllerProfile | None:
    """
        Returns a controller profile by key.
    """
    if key is None:
        return None
    return CONTROLLER_PROFILES.get(key, None)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def profile_for_config(config: tp.Any) -> ControllerProfile | None:
    """
        Resolves the profile of an existing controller configuration.

        This is what allows loading a model without ever asking the user for a controller type.
    """
    for profile in CONTROLLER_PROFILES.values():
        config_cls = profile.config_cls
        if config_cls is not None and isinstance(config, config_cls):
            return profile
    return None

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

BRAIN_PROFILE = register_controller_profile(
    ControllerProfile(
        key='brain',
        label='Brain',
        icon=icons.BRAIN,
        summary='Coordinates neuron pools and interfaces. Every module reads the previous state from a shared cache.',
        controller_name='Brain',
        palette_namespaces=(RegistryNamespace.Neurons, RegistryNamespace.Interfaces),
        atomic_namespaces=(RegistryNamespace.Neurons,),
        cache_based=True,
    )
)

NEURON_PROFILE = register_controller_profile(
    ControllerProfile(
        key='neuron',
        label='Neuron',
        icon=icons.NEURON,
        summary='Assembles somas, synapses, delays and plasticity rules into a single pool of neurons.',
        controller_name='Neuron',
        palette_namespaces=(RegistryNamespace.Components, RegistryNamespace.Interfaces),
        atomic_namespaces=(RegistryNamespace.Components,),
        import_namespaces=(RegistryNamespace.Neurons,),
        model_namespace=RegistryNamespace.Neurons,
        cache_based=False,
    )
)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
