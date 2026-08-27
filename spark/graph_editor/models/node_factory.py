#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import logging
import typing as tp
import dataclasses as dc
import enum
import importlib
import spark.core.utils as utils
from spark.core.module import SparkModule
from spark.core.registry import RegistryEntry, Registry, REGISTRY, RegistryNamespace
from spark.graph_editor.models.node_model import NodeModel, ComponentNodeModel, InterfaceNodeModel, ControllerNodeModel
from spark.graph_editor.models.port_model import PortModel
from spark.core.config import SparkConfig

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class NodeFactory:
    """
        Builds NodeModel classes from registry entries.
    """

    @staticmethod
    def create_node_from_registry(entry: RegistryEntry, base_node_cls: type[NodeModel]) -> type[NodeModel]:
        """
            Creates a NodeModel class from a registry entry.
        """
        node_model_cls = type(
            entry.name,
            (base_node_cls,),
            {
                '_cls': entry.get_cls(),
            } 
        )
        return node_model_cls

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class NodeRegistry:
    """
        Graph Editor Registry for node models.
    """

    NAMESPACE_BASE_MODEL = {
        RegistryNamespace.Components: ComponentNodeModel,
        RegistryNamespace.Interfaces: InterfaceNodeModel,
        RegistryNamespace.Neurons: ControllerNodeModel,
    }

    def __init__(self,) -> None:
        super().__init__()
        self._registry: dict[type, type[NodeModel]] = {}
        self._namespaces: dict[type, RegistryNamespace] = {}

        for namespace, base_model in self.NAMESPACE_BASE_MODEL.items():
            for _, entry in getattr(REGISTRY, namespace.name).items():
                self._map(entry, namespace, base_model)

    def _map(self, entry: RegistryEntry, namespace: RegistryNamespace, base_model: type[NodeModel]) -> type[NodeModel] | None:
        """
            Builds the node model of an entry.
        """
        if len(entry.path) > 0 and entry.path[0].lower() == 'controller':
            return None
        cls = entry.get_cls()
        self._registry[cls] = NodeFactory.create_node_from_registry(entry, base_model)
        self._namespaces[cls] = namespace
        return self._registry[cls]

    def _adopt(self, node_cls: type) -> type[NodeModel] | None:
        """
            Builds the node model of a class after initialization.
        """
        for namespace, base_model in self.NAMESPACE_BASE_MODEL.items():
            entry = getattr(REGISTRY, namespace.name).get_by_cls(node_cls)
            if entry:
                return self._map(entry, namespace, base_model)
        return None

    def get(self, node_cls: type) -> type[NodeModel] | None:
        node_model_cls = self._registry.get(node_cls, None)
        return node_model_cls if node_model_cls is not None else self._adopt(node_cls)

    def get_namespace(self, node_cls: type) -> RegistryNamespace | None:
        """
            Returns the registry namespace of a node.
        """
        if node_cls not in self._namespaces:
            self._adopt(node_cls)
        return self._namespaces.get(node_cls, None)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

NODE_REGISTRY = NodeRegistry()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################