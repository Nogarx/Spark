#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.graph_model import GraphModel
    from spark.graph_editor.models.inspector_model import ConfigValueNode

import copy
import logging
import typing as tp
from shiboken6 import isValid
from PySide6.QtGui import QUndoCommand

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ChangeConfigValueCommand(QUndoCommand):

    def __init__(
            self, 
            graph_model: GraphModel, 
            path: list[str], 
            old_value: tp.Any, 
            new_value: tp.Any, 
            node: ConfigValueNode, 
            description: str = 'Change Config Value'
        ) -> None:
        super().__init__(description)
        self.graph_model = graph_model
        self.path = path
        self.old_value = copy.deepcopy(old_value)
        self.new_value = copy.deepcopy(new_value)
        self.node = node
        self._is_first_run = True

    def id(self) -> int:
        # Id built from the node and the config path, so consecutive edits of one field are merged.
        path_str = '/'.join(str(p) for p in self.path)
        return 200 + hash(path_str) % 10000

    def mergeWith(self, command) -> bool:
        if command.id() != self.id():
            return False
        # Merging keeps the original old_value and takes the newer value.
        self.new_value = copy.deepcopy(command.new_value)
        return True

    def undo(self) -> None:
        # Revert the Python backend.
        self.graph_model.set_node_config_value(self.path, self.old_value)
        # Propagate through the inheritance tree.
        self.graph_model.update_inherited_value(self.path, self.old_value)
        
        # Revert the UI state model.
        if isValid(self.node):
            self.node.value = self.old_value
        logger.info(f'Undo config change at {self.path}')

    def redo(self) -> None:
        # Update the Python backend.
        self.graph_model.set_node_config_value(self.path, self.new_value)
        # Propagate through the inheritance tree.
        self.graph_model.update_inherited_value(self.path, self.new_value)
        
        # Update the UI state model.
        if isValid(self.node):
            self.node.value = self.new_value
            
        if self._is_first_run:
            self._is_first_run = False
        else:
            logger.info(f'Redo config change at {self.path} to {self.new_value}')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ToggleInheritanceCommand(QUndoCommand):

    def __init__(
            self, 
            graph_model: GraphModel, 
            path: list[str], 
            old_state: bool, 
            new_state: bool, 
            node: ConfigValueNode, 
            description: str = 'Toggle Inheritance'
        ) -> None:
        super().__init__(description)
        self.graph_model = graph_model
        self.path = path
        self.old_state = old_state
        self.new_state = new_state
        self.node = node
        self.old_child_values = {}

    def undo(self) -> None:
        if isValid(self.node):
            self.node.is_inherited = self.old_state
        self.graph_model.toggle_inheritance(self.path, self.old_state)
        
        # Restore the child values that were overridden.
        if self.new_state:
            for child_path_tuple, old_val in self.old_child_values.items():
                child_path = list(child_path_tuple)
                self.graph_model.set_node_config_value(child_path, old_val, force=True)


        logger.info(f'Undo inheritance toggle at {self.path}')

    def redo(self) -> None:
        if isValid(self.node):
            self.node.is_inherited = self.new_state
        self.graph_model.toggle_inheritance(self.path, self.new_state)
        
        if self.new_state:
            # Snapshot the values that are about to be overwritten by the cascade.
            for child_path in self.graph_model.get_inheritance_children(self.path):
                path_key = tuple(child_path)
                if path_key not in self.old_child_values:
                    old_val = self.graph_model.get_node_config_value(child_path)
                    self.old_child_values[path_key] = copy.deepcopy(old_val)
            if isValid(self.node):
                self.graph_model.update_inherited_value(self.path, self.node.value)
        logger.info(f'Redo inheritance toggle at {self.path}')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################