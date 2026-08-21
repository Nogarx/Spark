#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.graph_model import GraphModel
    from spark.graph_editor.models.edge_model import EdgeModel
    from spark.graph_editor.models.node_model import NodeModel

import logging
from PySide6.QtGui import QUndoCommand

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class MoveNodeCommand(QUndoCommand):

    def __init__(self, node_model: NodeModel, old_pos: tuple[float, float], new_pos: tuple[float, float], description: str = 'Move Node') -> None:
        super().__init__(description)
        self.node_model = node_model
        self.old_pos = old_pos
        self.new_pos = new_pos

    def undo(self) -> None:
        self.node_model.pos = self.old_pos
        logger.info(f'Undo: Moved node "{self.node_model.name}" to {self.old_pos}')

    def redo(self) -> None:
        self.node_model.pos = self.new_pos
        logger.info(f'Moved node "{self.node_model.name}" to {self.new_pos}')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class AddNodeCommand(QUndoCommand):

    def __init__(self, graph_model: GraphModel, node_model: NodeModel, description: str = 'Add Node') -> None:
        super().__init__(description)
        self.graph_model = graph_model
        self.node_model = node_model

    def undo(self) -> None:
        self.graph_model.remove_node(self.node_model)
        logger.info(f'Undo: Removed node "{self.node_model.name}"')

    def redo(self) -> None:
        self.graph_model.add_node(self.node_model)
        logger.info(f'Added node "{self.node_model.name}"')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RemoveNodeCommand(QUndoCommand):

    def __init__(self, graph_model: GraphModel, node_model: NodeModel, description: str ='Remove Node') -> None:
        super().__init__(description)
        self.graph_model = graph_model
        self.node_model = node_model
        self.associated_edges = []
        
        # Identify all edges connected to this node
        for port in node_model.get_all_ports():
            for edge in port.edges:
                if edge not in self.associated_edges:
                    self.associated_edges.append(edge)

    def undo(self) -> None:
        self.graph_model.add_node(self.node_model)
        for edge in self.associated_edges:
            self.graph_model.add_edge(edge)
        logger.info(f'Undo: Restored node "{self.node_model.name}" and {len(self.associated_edges)} edges')

    def redo(self) -> None:
        # Edges are automatically removed by graph_model.remove_node
        self.graph_model.remove_node(self.node_model)
        logger.info(f'Removed node "{self.node_model.name}"')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class AddEdgeCommand(QUndoCommand):

    def __init__(self, graph_model: GraphModel, edge_model: EdgeModel, description: str = 'Add Edge') -> None:
        super().__init__(description)
        self.graph_model = graph_model
        self.edge_model = edge_model

    def undo(self) -> None:
        self.graph_model.remove_edge(self.edge_model)
        logger.info(f'Undo: Removed edge between "{self.edge_model.source_port.name}" and "{self.edge_model.target_port.name}"')

    def redo(self) -> None:
        self.graph_model.add_edge(self.edge_model)
        logger.info(f'Added edge between "{self.edge_model.source_port.name}" and "{self.edge_model.target_port.name}"')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RemoveEdgeCommand(QUndoCommand):

    def __init__(self, graph_model: GraphModel, edge_model: EdgeModel, description: str ='Remove Edge') -> None:
        super().__init__(description)
        self.graph_model = graph_model
        self.edge_model = edge_model

    def undo(self) -> None:
        self.graph_model.add_edge(self.edge_model)
        logger.info(f'Undo: Restored edge between "{self.edge_model.source_port.name}" and "{self.edge_model.target_port.name}"')

    def redo(self) -> None:
        self.graph_model.remove_edge(self.edge_model)
        logger.info(f'Removed edge between "{self.edge_model.source_port.name}" and "{self.edge_model.target_port.name}"')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ChangeEdgeWaypointsCommand(QUndoCommand):

    def __init__(self, edge_model: EdgeModel, old_waypoints: list[tuple], new_waypoints: list[tuple], description: str = 'Change Pipe Route') -> None:
        super().__init__(description)
        self.edge_model = edge_model
        self.old_waypoints = old_waypoints
        self.new_waypoints = new_waypoints

    def undo(self) -> None:
        self.edge_model.waypoints = list(self.old_waypoints)
        logger.info(f'Undo change waypoints for edge')

    def redo(self) -> None:
        self.edge_model.waypoints = list(self.new_waypoints)
        logger.info(f'Redo change waypoints for edge')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RenameNodeCommand(QUndoCommand):

    def __init__(self, node_model: NodeModel, old_name: str, new_name: str, description: str = 'Rename Node') -> None:
        super().__init__(description)
        self.node_model = node_model
        self.old_name = old_name
        self.new_name = new_name
        self._is_first_run = True
        
    def id(self) -> int:
        return 100 + hash(self.node_model.id) % 10000

    def mergeWith(self, command) -> bool:
        if command.id() != self.id():
            return False
        # Merge by updating the new name, keeping the original old_name
        self.new_name = command.new_name
        return True

    def undo(self) -> None:
        from shiboken6 import isValid
        if isValid(self.node_model):
            self.node_model.name = self.old_name
            logger.info(f'Undo rename node to "{self.old_name}"')

    def redo(self) -> None:
        from shiboken6 import isValid
        if isValid(self.node_model):
            self.node_model.name = self.new_name
            if self._is_first_run:
                self._is_first_run = False
            else:
                logger.info(f'Redo rename node to "{self.new_name}"')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################