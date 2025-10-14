#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import networkx as nx
from math import prod
from typing import Dict, List, Tuple
from PySide6 import QtCore
from NodeGraphQt import NodeGraph, Port, BaseNode
from NodeGraphQt.widgets.viewer import NodeViewer
from spark.core.specs import PortMap
from spark.core.shape import Shape
from spark.graph_editor.models.nodes import SourceNode, SinkNode, AbstractNode, module_to_nodegraph
from spark.graph_editor.models.graph_menu_tree import HierarchicalMenuTree

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class SparkNodeViewer(NodeViewer):
    """
        Custom reimplementation of NodeViwer use to solve certain visual bugs.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ignore_next_click_flag = False

    def _ignore_next_click_event(self, state: bool):
        self._ignore_next_click_flag = state

    def mousePressEvent(self, event):
        # If the flag is set, we consume the event. This is used to avoid certain visual bugs.
        if self._ignore_next_click_flag:
            self._ignore_next_click_flag = False
            return
        super().mousePressEvent(event)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkNodeGraph(NodeGraph):
    """
        NodeGraphQt object for building/managing Spark models.
    """

    stateChanged = QtCore.Signal() 
    _is_modified = False

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs, viewer=SparkNodeViewer())
        # Enable recurrence
        self.set_acyclic(False)

        self._node_registry: Dict[str, str] = {}
        # TODO: Current validation logic is to rigid, but there may be a workaround.
        # The easy work around is to validate the connection after an attempt has been made and refute it if it is not valid.
        # Add a callback to validate logic.
        self.port_connected.connect(self._on_port_connected)
        self.port_disconnected.connect(self._on_port_disconnected)
        # Callbacks to make NODE_NAME unique (required for readable model files)
        self.viewer().node_name_changed.connect(self._on_node_name_changed)
        # TODO: Undo stack need to be properly modified to accomodate custom node attributes.
        # On state changed.
        self.node_created.connect(self._on_state_changed)
        self.nodes_deleted.connect(self._on_state_changed)
        self.port_connected.connect(self._on_state_changed)
        self.port_disconnected.connect(self._on_state_changed)
        self.property_changed.connect(self._on_state_changed)
        self.data_dropped.connect(self._on_state_changed)
        self.session_changed.connect(self._on_state_changed)

    context_menu_prompt = QtCore.Signal(object, object)

    def _on_state_changed(self):
        self._is_modified = True
        self.stateChanged.emit()

    def ignore_next_click_event(self, state: bool = True):
        self.viewer()._ignore_next_click_event(state)

    def name_exist(self, name: str):
        return name in self._node_registry.values()
    
    def update_node_name(self, id: str, name: str):
        if self.name_exist(name):
            return
        self.get_node_by_id(id).set_name(name)
        self._node_registry[id] = name

    def _on_output_shape_update(self, output_port: Port):
        connected_ports: List[Port] = output_port.connected_ports()
        for port in connected_ports:
            self._update_port_specs(port)

    def _on_port_connected(self, input_port: Port, output_port: Port):
        # Delete invalid connections
        input_node: AbstractNode = input_port.node()
        input_payload_type = input_node.input_specs[input_port.name()].payload_type
        output_node: AbstractNode = output_port.node()
        output_payload_type = output_node.output_specs[output_port.name()].payload_type
        if input_payload_type != output_payload_type: 
            input_port.disconnect_from(output_port, push_undo=False, emit_signal=False)
            return

    def _on_port_disconnected(self, input_port: Port, output_port: Port):
        # Get nodes.
        input_node: AbstractNode = input_port.node()
        output_node: AbstractNode = output_port.node()

    def _update_port_specs(self, input_port: Port):
        return
        # Get node
        input_node: AbstractNode = input_port.node()
        # Get connected ports
        port_map_list = input_node.input_specs[input_port.name()].port_maps
        if len(port_map_list) == 0:
            shape = None
        elif len(port_map_list) == 1:
            port_map = port_map_list[0]
            output_node: AbstractNode = self.get_node_by_id(port_map.origin)
            port_shape = output_node.output_specs[port_map.port].shape
            shape = Shape(port_shape) if port_shape else None
        else:
            shape = 0
            for port_map in port_map_list:
                output_node: AbstractNode = self.get_node_by_id(port_map.origin)
                port_shape = output_node.output_specs[port_map.port].shape
                port_shape = Shape(output_node.output_specs[port_map.port].shape) if port_shape else None
                shape = shape + prod(port_shape) if port_shape else shape
            shape = Shape(shape)
        input_node.input_specs[input_port.name()].shape = shape

    def _on_node_name_changed(self, node_id: str, node_name: str):
        self._node_registry[node_id] = node_name

    def create_node(self, node_type, name=None, selected=True, color=None, text_color=None, pos=None, push_undo=True) -> AbstractNode:
        node: BaseNode = super().create_node(node_type, name, selected, color, text_color, pos, push_undo)
        self._node_registry[node.id] = node.NODE_NAME
        return node
    
    def delete_node(self, node, push_undo=True):
        del self._node_registry[node.id]
        return super().delete_node(node, push_undo)

    def build_raw_graph(self,) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        all_nodes: List[BaseNode] = self.all_nodes()
        # Add nodes to the graph.
        for node in all_nodes:
            nx_graph.add_node(node.NODE_NAME, node_type=type(node).__name__)
        # Add edges
        for node in all_nodes:
            for input_port in node.connected_input_nodes().values():
                for input_node in input_port:
                    nx_graph.add_edge(input_node.NODE_NAME, node.NODE_NAME)
            for output_port in node.connected_output_nodes().values():
                for output_node in output_port:
                    nx_graph.add_edge(node.NODE_NAME, output_node.NODE_NAME)
        return nx_graph
    
    def get_nodes_by_map(self,) -> Tuple[List[AbstractNode], List[AbstractNode], List[AbstractNode]]:
        source_nodes = []
        sink_nodes = []
        internal_nodes = []
        for node in self.all_nodes():
            if isinstance(node, SourceNode):
                source_nodes.append(node)
            elif isinstance(node, SinkNode):
                sink_nodes.append(node)
            else:
                internal_nodes.append(node)
        return source_nodes, sink_nodes, internal_nodes

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################