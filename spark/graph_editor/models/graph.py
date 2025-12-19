#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import networkx as nx
from typing import Dict, List, Tuple
from PySide6 import QtCore
from NodeGraphQt import NodeGraph, Port, BaseNode
from NodeGraphQt.widgets.viewer import NodeViewer

from spark.core.specs import PortSpecs, PortMap, ModuleSpecs
from spark.nn.brain import BrainConfig
from spark.graph_editor.models.nodes import SourceNode, SinkNode, AbstractNode, SparkModuleNode
from spark.graph_editor.ui.console_panel import MessageLevel

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

    context_menu_prompt = QtCore.Signal(object, object)
    broadcast_message = QtCore.Signal(MessageLevel, str)
    stateChanged = QtCore.Signal(bool) 

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs, )#viewer=SparkNodeViewer())
        # Enable recurrence
        self.set_acyclic(False)
        self._is_modified = False
        self._node_registry: Dict[str, str] = {}
        # Add a callback to validate logic.
        self.port_connected.connect(self._on_port_connected)
        self.port_disconnected.connect(self._on_port_disconnected)
        # Callbacks to make NODE_NAME unique (required for readable model files)
        self.viewer().node_name_changed.connect(self._on_node_name_changed)
        # TODO: This doesnt really capture the is_modified status, but ro properly fix this issue we need to createa general undo/redo stack.
        self.node_created.connect(lambda: self._on_state_changed(True))
        self.nodes_deleted.connect(lambda: self._on_state_changed(True))
        self.port_connected.connect(lambda: self._on_state_changed(True))
        self.port_disconnected.connect(lambda: self._on_state_changed(True))
        self.property_changed.connect(lambda: self._on_state_changed(True))
        self.data_dropped.connect(lambda: self._on_state_changed(True))
        self.session_changed.connect(lambda: self._on_state_changed(True))
        self.node_selection_changed.connect(self._bugfix_on_node_selection_changed)
        self._prev_selection = []
    

    # NOTE: This is a simple workaround to fix a bug that happens when cliking an already selected node in the graph.
    def _bugfix_on_node_selection_changed(self, new_selection: list[AbstractNode], prev_selection: list[AbstractNode]) -> None:
        # Check if bug happend.
        if len(new_selection) == 0 and len(prev_selection) == 0:
            QtCore.QTimer.singleShot(1, self._bugfix_on_node_selection_changed_after_wait)
        else:
            # Cache previous selection.
            self._prev_selection = [n.id for n in new_selection]
    def _bugfix_on_node_selection_changed_after_wait(self) -> None:
        # Find selected node
        selected = None
        for node in self.all_nodes():
            if node.selected():
                selected = node.id
        # Remove selected from list
        try:
            self._prev_selection.remove(selected)
            # Broadcast correct selection
            self._on_node_selection_changed([selected], self._prev_selection)
        except:
            pass # Something weird happend
    

    def _on_state_changed(self, value: bool) -> None:
        self._is_modified = value
        self.stateChanged.emit(self._is_modified)

    #def ignore_next_click_event(self, state: bool = True) -> None:
    #    self.viewer()._ignore_next_click_event(state)

    def name_exist(self, name: str) -> bool:
        return name in self._node_registry.values()
    
    def update_node_name(self, id: str, name: str) -> None:
        if self.name_exist(name):
            return
        self.get_node_by_id(id).set_name(name)
        self._node_registry[id] = name

    def _on_output_shape_update(self, output_port: Port) -> None:
        connected_ports: List[Port] = output_port.connected_ports()
        for port in connected_ports:
            self._update_port_specs(port)

    def _on_port_connected(self, input_port: Port, output_port: Port) -> None:
        # Get node info
        input_node: AbstractNode = input_port.node()
        input_payload_type = input_node.input_specs[input_port.name()].payload_type
        output_node: AbstractNode = output_port.node()
        output_payload_type = output_node.output_specs[output_port.name()].payload_type

        # Update sink node specs
        if isinstance(input_node, SinkNode):
            output_specs = output_node.output_specs[output_port.name()]
            input_node.update_io_spec(
                'input', 
                'value', 
                payload_type=output_specs.payload_type,
                shape=output_specs.shape,
                dtype=output_specs.dtype,
            )
            return

        # Delete invalid connections
        if input_payload_type != output_payload_type: 
            input_port.disconnect_from(output_port, push_undo=False, emit_signal=False)
            self.broadcast_message.emit(
                MessageLevel.WARNING,
                f'Tried to connect {output_node.NODE_NAME}.{output_port.name()} to {input_node.NODE_NAME}.{input_port.name()} \
                but they have different payload types: {output_payload_type} vs {input_payload_type}.'
            )
            return
        

    def _on_port_disconnected(self, input_port: Port, output_port: Port) -> None:
        return
    
    def _update_port_specs(self, input_port: Port) -> None:
        return

    def _on_node_name_changed(self, node_id: str, node_name: str) -> None:
        self._node_registry[node_id] = node_name

    def create_node(self, node_type, name=None, selected=True, color=None, text_color=None, pos=None, push_undo=True) -> AbstractNode:
        node: BaseNode = super().create_node(node_type, name, selected, color, text_color, pos, push_undo)
        self._node_registry[node.id] = node.NODE_NAME
        # Graph automatically selects the new node. We need to trigger the event to keep it consistent.
        self._on_node_selection_changed([node.id], self._prev_selection)
        return node
    
    def delete_node(self, node, push_undo=True) -> None:
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

    # TODO: Improve node placent for configs without pos metadata
    def load_from_model(self, config: BrainConfig):
        node_names_to_ids = {}
        # Input nodes.
        for name, spec in config.input_map.items():
            pos = config.__graph_editor_metadata__.get(name, {}).get('pos', [0,0])
            node: SourceNode = self.create_node('spark.SourceNode', name=name, pos=pos)
            node.output_specs['value'].payload_type = spec.payload_type
            node.output_specs['value'].dtype = spec.dtype
            node.output_specs['value'].shape = spec.shape
            node.output_specs['value'].description = spec.description
            node_names_to_ids[name] = node.id
        # Output nodes.
        for name, dict_spec in config.output_map.items():
            pos = config.__graph_editor_metadata__.get(name, {}).get('pos', [0,0])
            node: SinkNode = self.create_node('spark.SinkNode', name=name, pos=pos)
            node.input_specs['value'].payload_type = dict_spec['spec'].payload_type
            node.input_specs['value'].dtype = dict_spec['spec'].dtype
            node.input_specs['value'].shape = dict_spec['spec'].shape
            node.input_specs['value'].description = dict_spec['spec'].description
            node_names_to_ids[name] = node.id
        # Module nodes.
        for name, spec in config.modules_map.items():
            pos = spec.config.__graph_editor_metadata__.get('pos', [0,0])
            node: SparkModuleNode = self.create_node(f'spark.{spec.module_cls.__name__}', name=name, pos=pos)
            node.node_config = spec.config
            node_names_to_ids[name] = node.id
        # Setup sink node connections
        for name, dict_spec in config.output_map.items():
            target_node: SinkNode = self.get_node_by_id(node_names_to_ids[name])
            target_port: Port = target_node.get_input('value')
            origin_node: AbstractNode = self.get_node_by_id(node_names_to_ids[dict_spec['input'].origin])
            origin_port: Port = origin_node.get_output(dict_spec['input'].port)
            target_port.connect_to(origin_port)
        # Setup module node connections
        for name, spec in config.modules_map.items():
            target_node: AbstractNode = self.get_node_by_id(node_names_to_ids[name])
            for port_name, port_maps in spec.inputs.items():
                target_port: Port = target_node.get_input(port_name)
                for p_map in port_maps:
                    if p_map.origin == '__call__':
                        origin_node: AbstractNode = self.get_node_by_id(node_names_to_ids[p_map.port])
                        origin_port: Port = origin_node.get_output('value')
                    else:
                        origin_node: AbstractNode = self.get_node_by_id(node_names_to_ids[p_map.origin])
                        origin_port: Port = origin_node.get_output(p_map.port)
                    target_port.connect_to(origin_port)
        # Clean stack
        self.clear_selection()
        self.clear_undo_stack()
        self.center_on()
        self._is_modified = False

    def validate_graph(self,) -> bool:
        """
            Simple graph validation. 
            
            Ensures that graph has a single connected component and at least one source and one sink is present in the model.
        """
        errors = []
        nx_graph = self.build_raw_graph()
        connected_components = list(nx.weakly_connected_components(nx_graph))
        # Check input/output exists.
        node_types = nx.get_node_attributes(nx_graph, 'node_type').values()
        has_input = 'SourceNode' in node_types
        has_output = 'SinkNode' in node_types
        if not has_input:
            errors.append(f'No source node was found.')
        if not has_output:
            errors.append(f'No sink node was found.')
        # Check number of connected components.
        if len(connected_components) > 1:
            errors.append(
                f'Multiple connected components are not currently supported but found {len(connected_components)} connected components.'
            )
        return errors

    def build_brain_config(self,) -> BrainConfig:
        """
            Build the model from the graph state.
        """
        input_map: dict[str, PortSpecs] = {}
        output_map: dict[str, dict] = {}
        modules_map: dict[str, ModuleSpecs] = {}
        io_nodes_metadata = {}
        for node in self.all_nodes():
            if isinstance(node, SourceNode):
                # Source nodes had a single output of the appropiate type
                input_map[node.NODE_NAME] = PortSpecs(
                    payload_type=node.output_specs['value'].payload_type,
                    shape=node.output_specs['value'].shape,
                    dtype=node.output_specs['value'].dtype,
                    description=node.output_specs['value'].description,
                )
                io_nodes_metadata[node.NODE_NAME] = {'pos': node.pos()}
            elif isinstance(node, SinkNode):
                # Sink nodes are virtual nodes, we need to get the origin of its input to resolve the output_map entry. 
                input_port: Port = node.get_input('value')
                # NOTE: Sink nodes are only allowed a single input.
                origin_port: Port = input_port.connected_ports()[0]
                origin_node: AbstractNode = origin_port.node()
                if isinstance(origin_node, SourceNode):
                    origin_name = '__call__'
                    port_name = origin_node.NODE_NAME
                else:
                    origin_name = origin_node.NODE_NAME
                    port_name = origin_port.name()
                # Create PortSpecs
                output_map[node.NODE_NAME] = {
                    'input': PortMap(origin=origin_name, port=port_name),
                    'spec': PortSpecs(
                        payload_type=node.input_specs['value'].payload_type,
                        shape=node.input_specs['value'].shape,
                        dtype=node.input_specs['value'].dtype,
                        description=node.input_specs['value'].description,
                    )
                }
                io_nodes_metadata[node.NODE_NAME] = {'pos': node.pos()}
            elif isinstance(node, SparkModuleNode):
                # Gather input maps
                inputs = {}
                for input_port in node.input_ports():
                    # Gather all incomming connections to the input port.
                    port_maps = []
                    for output_port in input_port.connected_ports():
                        origin_node: AbstractNode = output_port.node()
                        if isinstance(origin_node, SourceNode):
                            origin_name = '__call__'
                            port_name = origin_node.NODE_NAME
                        else:
                            origin_name = origin_node.NODE_NAME
                            port_name = output_port.name()
                        port_maps.append(PortMap(origin=origin_name, port=port_name))
                    inputs[input_port.name()] = port_maps
                # Build module spec
                node._update_graph_metadata()
                modules_map[node.NODE_NAME] = ModuleSpecs(
                    name = node.NODE_NAME,
                    module_cls = node.module_cls,
                    inputs = inputs,
                    config = node.node_config,
                )
        # Construct brain config.
        brain_config = BrainConfig(
            input_map=input_map, 
            output_map=output_map, 
            modules_map=modules_map
        )
        brain_config.__graph_editor_metadata__ = io_nodes_metadata
        return brain_config

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################