#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import typing as tp
from PySide6 import QtWidgets, QtCore, QtGui

from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG
from spark.graph_editor.models.graph import SparkNodeGraph
from spark.graph_editor.widgets.dock_panel import QDockPanel
from spark.graph_editor.models.nodes import AbstractNode, SparkModuleNode, BaseNode, SinkNode

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class NodesPanel(QDockPanel):
    """
        Dockable panel that shows a summary of the nodes in the graph.
    """

    def __init__(
            self, 
            node_graph: SparkNodeGraph, 
            name: str = 'Nodes', 
            parent: QtWidgets.QWidget = None, 
            **kwargs
        ) -> None:
        super().__init__(name, parent=parent, **kwargs)
        self.setMinimumWidth(GRAPH_EDITOR_CONFIG.nodes_panel_min_width)
        self._target_node = None
        self.node_list = QNodeList(node_graph, parent=self)
        self.setContent(self.node_list)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class QNodeList(QtWidgets.QWidget):
    """
        Constructs the nodes list associated with the SparkNodeGraph.
    """

    def __init__(self, node_graph: SparkNodeGraph, parent: QtWidgets.QWidget = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        # Graph reference
        self._node_graph = node_graph
        self._node_widget_map: dict[str, ListEntry] = {}
        self._current_selection: list[str] = []
        # Widget layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(8, 8, 8, 8))
        self.setLayout(layout)
        # Scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.layout().addWidget(scroll_area)
        scroll_area.setStyleSheet(
            f"""
                QScrollArea {{ 
                    background: {GRAPH_EDITOR_CONFIG.input_field_bg_color}; 
                    border: 1px solid {GRAPH_EDITOR_CONFIG.border_color};
                }}
            """
        )
        # Content widget
        self.content = QtWidgets.QWidget()
        self.content.setStyleSheet(
            """
                QWidget { 
                    background: transparent; 
                }
            """
        )
        scroll_area.setWidget(self.content)
        content_layout = QtWidgets.QVBoxLayout(self.content)
        content_layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        content_layout.setSpacing(0)
        # Setup layout
        self._setup_layout()
        # Events
        self._node_graph.node_selection_changed.connect(self.on_selection_update)
        self._node_graph.viewer().node_name_changed.connect(self.on_node_name_update)
        self._node_graph.nodes_deleted.connect(self.on_nodes_deletion)
        self._node_graph.node_created.connect(self.on_node_created)
        self._node_graph.session_changed.connect(self.on_session_changed)

    def addWidget(self, widget: QtWidgets.QWidget) -> None:
        """
            Add a widget to the central content widget's layout.
        """
        widget.installEventFilter(self)
        self.content.layout().addWidget(widget)

    def _setup_layout(self,) -> None:
        for node in self._node_graph.all_nodes():
            self._add_widget_from_node(node)

    def _add_widget_from_node(self, node: AbstractNode) -> None:
        # Add widget
        entry = ListEntry(node.id, node.NODE_NAME, parent=self)
        entry.clicked.connect(self.on_entry_clicked)
        self.addWidget(entry)
        # Add to registry
        self._node_widget_map[node.id] = entry

    def on_nodes_deletion(self, nodes_ids: list[str]) -> None:
        # NOTE: There is another method also deleting nodes so they may not exist when this event gets triggered.
        # on_session_changed handles the extreme case.
        for n in nodes_ids:
            try:
                widget = self._node_widget_map.pop(n)
                widget.deleteLater()
            except:
                pass

    def on_session_changed(self,) -> None:
        # Clear list.
        for widget in self._node_widget_map.values():
            try:
                widget.deleteLater()
            except:
                pass
        # Repopulate list
        self._node_widget_map = {}
        for node in self._node_graph.all_nodes():
            self._add_widget_from_node(node)

    def on_node_created(self, node: AbstractNode) -> None:
        self._add_widget_from_node(node)

    def on_entry_clicked(self, node_id: str) -> None:
        # Check if selected to avoid wasted broadcasting
        node: AbstractNode = self._node_graph.get_node_by_id(node_id)
        if node:
            if not node.selected():
                # Select this node
                node.set_selected(True)
                # Unselect other nodes
                for node_id in self._current_selection:
                    self._node_graph.get_node_by_id(node_id).set_selected(False)
                # Broadcast selection
                self._node_graph._on_node_selection_changed([node.id], self._current_selection)
            else:
                if len(self._current_selection) > 1:
                    # Unselect other nodes
                    for node_id in self._current_selection:
                        if node.id == node_id:
                            continue
                        self._node_graph.get_node_by_id(node_id).set_selected(False)
                    # Broadcast selection
                    self._node_graph._on_node_selection_changed([node.id], self._current_selection)

    def on_selection_update(self, new_selection: list[AbstractNode], prev_selection: list[AbstractNode]) -> None:
        # Remove previous selection
        for node in prev_selection:
            if node:
                self._node_widget_map[node.id].set_selection(False)
        # Add new slection
        self._current_selection = []
        for node in new_selection:
            if node:
                self._node_widget_map[node.id].set_selection(True)
            self._current_selection.append(node.id)

    def on_node_name_update(self, node_id: str, name: str) -> None:
        self._node_widget_map[node_id].label.setText(name)
    

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ListEntry(QtWidgets.QWidget):
    """
        QWidget associated with each entry in the list.
    """

    clicked = QtCore.Signal(str)

    def __init__(self, node_id: str, node_name: str, parent: QtWidgets.QWidget = None) -> None:
        super().__init__(parent=parent)
        self._node_id = node_id
        self._selected = False
        # Styles
        self._hover_style = f"""
            background-color: {GRAPH_EDITOR_CONFIG.hover_color};
        """
        self._selected_style = f"""
            background-color: {GRAPH_EDITOR_CONFIG.selected_color};
        """
        self._non_selected_style = f"""
            background-color: {GRAPH_EDITOR_CONFIG.input_field_bg_color};
        """
        self._current_style = self._non_selected_style

        # Add layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        # Add QLineEdit
        self.label = QtWidgets.QLabel(node_name, parent=self)
        self.label.setFixedHeight(24)
        self.label.setStyleSheet(
            f"""
                color: {GRAPH_EDITOR_CONFIG.default_font_color};
                font-size: {GRAPH_EDITOR_CONFIG.small_font_size}px;
                padding: 2px
            """
        )
        layout.addWidget(self.label)
        # Finalize
        self.setLayout(layout)

    def on_name_update(self, name: str) -> None:
        self.label.setText(name)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        # Set flag for simple clicks
        self._clicked_inside = True

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        release_pos = event.position().toPoint()
        # Check if the press started here and the release is still inside
        if self._clicked_inside and self.rect().contains(release_pos):
            # Emit clicked, let parent handle the updates
            self.clicked.emit(self._node_id)
        # Set flag off for simple clicks
        self._clicked_inside = False

    def enterEvent(self, event: QtCore.QEvent) -> None:
        self.setStyleSheet(self._hover_style)
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.setStyleSheet(self._current_style)
        super().leaveEvent(event)

    def set_selection(self, value: bool) -> None:
        if value:
            self._current_style = self._selected_style
            self.setStyleSheet(self._current_style)
        else:
            self._current_style = self._non_selected_style
            self.setStyleSheet(self._current_style)
        self._selected = value

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################