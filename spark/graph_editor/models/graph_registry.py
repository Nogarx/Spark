#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import enum
import logging
import typing as tp
import dataclasses as dc
import spark.core.utils as utils
from spark.core.registry import REGISTRY, Registry, SubRegistry

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class GraphEditorRegistryNamespace(enum.Enum):
    Components = enum.auto()
    Initializers = enum.auto()
    Interfaces = enum.auto()
    Neurons = enum.auto()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class GraphEditorRegistry(Registry):
    """
        Generic registry implementation.
    """

    if tp.TYPE_CHECKING:
        Components: SubRegistry
        Initializers: SubRegistry
        Interfaces: SubRegistry
        Neurons: SubRegistry

    def __init__(self,) -> None:
        super().__init__()
        self._raw_registry = utils.TwoKeyDict({r: {} for r in GraphEditorRegistryNamespace._member_map_.values()})
        self._registry = utils.TwoKeyDict({r: {} for r in GraphEditorRegistryNamespace._member_map_.values()})


    # TODO: Refreshes the graph editor registry. Replace with a proper invalidation.
    def _rebuild_registry(self,) -> None:
        self._registry = utils.TwoKeyDict({r: {} for r in GraphEditorRegistryNamespace._member_map_.values()})
        self._populate_from_registry(REGISTRY.Components, GraphEditorRegistryNamespace.Components)
        self._populate_from_registry(REGISTRY.Initializers, GraphEditorRegistryNamespace.Initializers)
        self._populate_from_registry(REGISTRY.Interfaces, GraphEditorRegistryNamespace.Interfaces)
        self._populate_from_registry(REGISTRY.Neurons, GraphEditorRegistryNamespace.Neurons)

    def _populate_from_registry(self, subregistry: SubRegistry, namespace: GraphEditorRegistryNamespace) -> None:
        for name, entry in subregistry.items():
            self.register(
                namespace=namespace,
                name=name,
            )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

EDITOR_REGISTRY = GraphEditorRegistry()
EDITOR_REGISTRY._build()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################