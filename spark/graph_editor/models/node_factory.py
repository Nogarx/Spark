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
from spark.core.registry import RegistryEntry, Registry, REGISTRY
from spark.graph_editor.models.node_model import NodeModel, ComponentNodeModel, InterfaceNodeModel
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

    def __init__(self,) -> None:
        super().__init__()
        self._registry = {}

        # Map available models 
        for _, entry in REGISTRY.Components.items():
            # NOTE: Skip controllers. Need to be done more gracefully.
            if entry.path[0].lower() == 'controller':
                continue
            self._registry[entry.get_cls()] = NodeFactory.create_node_from_registry(entry, ComponentNodeModel)
        for _, entry in REGISTRY.Interfaces.items():
            self._registry[entry.get_cls()] = NodeFactory.create_node_from_registry(entry, InterfaceNodeModel)

    def get(self, node_cls: type) -> type[NodeModel] | None:
        return self._registry.get(node_cls, None)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Singleton
NODE_REGISTRY = NodeRegistry()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################