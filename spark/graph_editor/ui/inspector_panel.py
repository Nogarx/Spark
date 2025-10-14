#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import typing as tp
from PySide6 import QtWidgets, QtCore, QtGui

from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG
from spark.graph_editor.widgets.dock_panel import QDockPanel
from spark.graph_editor.widgets.inspector_idle import InspectorIdleWidget
from spark.graph_editor.widgets.node_config import QNodeConfig
from spark.graph_editor.models.nodes import AbstractNode

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InspectorPanel(QDockPanel):
    """
        Generic dockable panel.
    """

    def __init__(self, name: str = 'Inspector', parent: QtWidgets.QWidget = None, **kwargs) -> None:
        super().__init__(name, parent=parent, **kwargs)
        self._target_node = None

    def on_graph_selection_update(self, new_selection: list[AbstractNode], previous_selection: list[AbstractNode]) -> None:
        """
            Event handler for graph selections.
        """
        if len(new_selection) == 1:
            self.set_node(new_selection[0])
        else: 
            self.set_node(None)

    def set_node(self, node: AbstractNode | None):
        """
            Sets the node to be inspected. If the node is None, it clears the inspector.
            
            Input:
                node: The node to inspect, or None to clear.
        """
        if self._target_node is node:
            return
        self._target_node = node
        
        if self._target_node:
            self.clearWidgets()
            self.setContent(
                QNodeConfig(self._target_node)
            )
        else:
            self.setContent(
                InspectorIdleWidget()
            )

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################