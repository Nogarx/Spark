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
        Utility class to create NodeModel instances from Registry entries.
    """

    @staticmethod
    def create_node_from_registry(entry: RegistryEntry, base_node_cls: type[NodeModel]) -> type[NodeModel]:
        """
            Creates and populates a NodeModel based on a RegistryEntry.
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

    # Node model base class used for each registry namespace.
    NAMESPACE_BASE_MODEL = {
        RegistryNamespace.Components: ComponentNodeModel,
        RegistryNamespace.Interfaces: InterfaceNodeModel,
        RegistryNamespace.Neurons: ControllerNodeModel,
    }

    def __init__(self,) -> None:
        super().__init__()
        self._registry: dict[type, type[NodeModel]] = {}
        self._namespaces: dict[type, RegistryNamespace] = {}

        # Map available models
        for namespace, base_model in self.NAMESPACE_BASE_MODEL.items():
            for _, entry in getattr(REGISTRY, namespace.name).items():
                # NOTE: Controllers are not placed as plain modules, they are the graph itself.
                if len(entry.path) > 0 and entry.path[0].lower() == 'controller':
                    continue
                cls = entry.get_cls()
                self._registry[cls] = NodeFactory.create_node_from_registry(entry, base_model)
                self._namespaces[cls] = namespace

    def get(self, node_cls: type) -> type[NodeModel] | None:
        return self._registry.get(node_cls, None)

    def get_namespace(self, node_cls: type) -> RegistryNamespace | None:
        """
            Registry namespace a node class was created from.
        """
        return self._namespaces.get(node_cls, None)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Singleton
NODE_REGISTRY = NodeRegistry()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################