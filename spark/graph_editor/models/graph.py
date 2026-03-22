#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import enum
import networkx as nx
from typing import Dict, List, Tuple
from PySide6 import QtCore, QtGui, QtWidgets
from NodeGraphQt import NodeGraph, Port, BaseNode
from NodeGraphQt.widgets.viewer import NodeViewer

from spark.core.registry import REGISTRY
from spark.core.specs import PortSpecs, PortMap, ModuleSpecs
from spark.nn.controllers.base import ControllerConfig
from spark.nn.controllers.neuron import NeuronConfig
from spark.nn.controllers.brain import BrainConfig
from spark.graph_editor.models.nodes import SourceNode, SinkNode, AbstractNode, SparkModuleNode, module_to_nodegraph, neuron_to_nodegraph
from spark.graph_editor.ui.console_panel import MessageLevel
from spark.graph_editor.style.painter import DEFAULT_PALETTE, PortColorStyle

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ControllerType(enum.IntFlag):
    BRAIN = 0b0
    NEURON = 0b1

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkNodeViewer(NodeViewer):
    """
        Custom reimplementation of NodeViwer use to solve certain bugs.
    """

    # BUG: Patch because NodeGraph keeps overriding the Save shortcut >:|
    BUGFIX_on_save = QtCore.Signal()

    def __init__(self, parent=None, undo_stack=None):
        super().__init__(parent, undo_stack)

    def focusInEvent(self, event):
        QtWidgets.QGraphicsView.focusInEvent(self, event)

    def focusOutEvent(self, event):
        QtWidgets.QGraphicsView.focusOutEvent(self, event)

    def keyPressEvent(self, event):
        """
            Explicitly ignore Ctrl+S so it propagates to the Main Window.
        """
        if event.matches(QtGui.QKeySequence.Save):
            self.BUGFIX_on_save.emit()

            # We ignore this specific event. Qt will then pass it 
            # to the parent widget (your Main Window).
            event.ignore()
            return

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkNodeGraph(NodeGraph):
    """
        NodeGraphQt object for building/managing Spark models.
    """

    context_menu_prompt = QtCore.Signal(object, object)
    broadcast_message = QtCore.Signal(MessageLevel, str)
    on_update = QtCore.Signal(bool) 

    def __init__(self, controller_type: ControllerType, parent=None, **kwargs) -> None:
        super().__init__(parent, **kwargs, viewer=SparkNodeViewer())
        # Controller type
        self._controller_type = controller_type
        # Register nodes
        self._register_nodes()
        # Enable recurrence
        self.set_acyclic(False)
        self._is_dirty = False
        self._node_registry: Dict[str, str] = {}
        # Add a callback to validate logic.
        self.port_connected.connect(self._on_port_connected)
        self.port_disconnected.connect(self._on_port_disconnected)
        # Callbacks to make NODE_NAME unique (required for readable model files)
        self.viewer().node_name_changed.connect(self._on_node_name_changed)
        # TODO: This doesnt really capture the is_modified status, but ro properly fix this issue we need to createa general undo/redo stack.
        self.node_created.connect(lambda: self._on_update(True))
        self.nodes_deleted.connect(lambda: self._on_update(True))
        self.port_connected.connect(lambda: self._on_update(True))
        self.port_disconnected.connect(lambda: self._on_update(True))
        self.property_changed.connect(lambda: self._on_update(True))
        self.data_dropped.connect(lambda: self._on_update(True))
        self.session_changed.connect(lambda: self._on_update(True))
        self._viewer.moved_nodes.connect(lambda: self._on_update(True))
        self.node_selection_changed.connect(self._bugfix_on_node_selection_changed)
        self._prev_selection = []
        
    
    def _register_nodes(self,):
        # Register source node model.
        self.register_node(SourceNode)
        # Register sink node model.
        self.register_node(SinkNode)
        # Register module node models.
        for key, entry in REGISTRY.MODULES.items():
            nodegraph_cls = module_to_nodegraph(entry)
            self.register_node(nodegraph_cls)
        # Register module node models.
        for key, entry in REGISTRY.NEURONS.items():
            nodegraph_cls = neuron_to_nodegraph(entry)
            self.register_node(nodegraph_cls)

    def set_controller_type(self, controller_type: ControllerType):
        if controller_type != self._controller_type:
            self._controller_type = controller_type
            msg_controller_type = 'Brain' if controller_type == ControllerType.BRAIN else 'Neuron'
            self.broadcast_message.emit(
                MessageLevel.INFO,
                f'Controller type updated to: "{msg_controller_type}".'
            )
    
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
    

    def _on_update(self, value: bool) -> None:
        self._is_dirty = value
        self.on_update.emit(self._is_dirty)

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

    def _on_port_connected(self, target_port: Port, origin_port: Port) -> None:
        # Get node info
        target_node: AbstractNode = target_port.node()
        origin_node: AbstractNode = origin_port.node()
        # Get port specs
        target_port_name = target_port.name()
        if target_port_name in target_node.input_specs:
            target_port_specs = target_node.input_specs[target_port_name]
        else:
            # NOTE: Source/sink nodes should not be able to reach here
            target_port_specs = target_node.property_specs[target_port_name]

        origin_port_name = origin_port.name()
        if origin_port_name in origin_node.output_specs:
            origin_port_specs = origin_node.output_specs[origin_port_name]
        else:
            # NOTE: Source/sink nodes should not be able to reach here
            origin_port_specs = origin_node.property_specs[origin_port_name]

        # Update source/sink node specs dinamically.
        if isinstance(target_node, SinkNode):
            target_node.update_io_spec(
                'input', 
                'value', 
                payload_type=origin_port_specs.payload_type,
                shape=origin_port_specs.shape,
                dtype=origin_port_specs.dtype,
            )
            # Update port painter fn
            target_port.view.set_painter(DEFAULT_PALETTE(origin_port_specs.payload_type.__name__, color_style=PortColorStyle.DEFAULT)) 
            return
        elif isinstance(origin_node, SourceNode) and len(origin_node.connected_output_nodes()[origin_port]) == 1:
            origin_node.update_io_spec(
                'output', 
                'value', 
                payload_type=target_port_specs.payload_type,
                shape=target_port_specs.shape,
                dtype=target_port_specs.dtype,
            )
            # Update port painter fn
            origin_port.view.set_painter(DEFAULT_PALETTE(target_port_specs.payload_type.__name__, color_style=PortColorStyle.DEFAULT)) 
            return

        # Delete invalid connections
        if origin_port_specs.payload_type != target_port_specs.payload_type: 
            target_port.disconnect_from(origin_port, push_undo=False, emit_signal=False)
            self.broadcast_message.emit(
                MessageLevel.WARNING,
                f'Tried to connect {origin_node.NODE_NAME}.{origin_port_name} to {target_node.NODE_NAME}.{target_port_name} '
                f'but they have different payload types: {origin_port_specs.payload_type} vs {target_port_specs.payload_type}.'
            )
            return
        
    def _on_port_disconnected(self, input_port: Port, output_port: Port) -> None:
        return
    
    def _update_port_specs(self, input_port: Port) -> None:
        return

    def _on_node_name_changed(self, node_id: str, node_name: str) -> None:
        self._node_registry[node_id] = node_name

    def create_node(
            self, 
            node_type: str,
            name: str = None, 
            selected: bool = True, 
            color: str | None = None, 
            text_color: str | None = None, 
            pos: list | tuple | None = None, 
            push_undo: bool = True, 
            select_node: bool = True
        ) -> AbstractNode:
        node: BaseNode = super().create_node(node_type, name, selected, color, text_color, pos, push_undo)
        self._node_registry[node.id] = node.NODE_NAME
        # Graph automatically selects the new node. This is not really ideal always.
        if select_node:
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
    def load_from_model(self, config: ControllerConfig):
        if isinstance(config, BrainConfig):
            self.set_controller_type(ControllerType.BRAIN)
        elif isinstance(config, NeuronConfig):
            self.set_controller_type(ControllerType.NEURON)
        else:
            raise TypeError(
                f'Unknown controller type: "{type(config).__name__}"'
            )
        node_names_to_ids = {}
        # IO nodes.
        for name, spec in config.__graph_editor_metadata__.items():
            #node_cls = SourceNode if spec['type'] == 'source' else SinkNode
            node_cls = 'spark.SourceNode' if spec['type'] == 'source' else 'spark.SinkNode'
            node_attr_label = 'output_specs' if spec['type'] == 'source' else 'input_specs'
            pos = spec.get('pos', [0,0])
            node: SourceNode = self.create_node(node_cls, name=name, pos=pos, select_node=False)
            attr_spec = getattr(node, node_attr_label)
            attr_spec['value'].payload_type = spec['spec'].payload_type
            attr_spec['value'].dtype = spec['spec'].dtype
            attr_spec['value'].shape = spec['spec'].shape
            attr_spec['value'].description = spec['spec'].description
            node_names_to_ids[name] = node.id
        # Module nodes.
        for module_spec in config.modules_specs:
            pos = module_spec.config.__graph_editor_metadata__.get('pos', [0,0])
            node: SparkModuleNode = self.create_node(f'spark.{module_spec.module_cls.__name__}', name=module_spec.name, pos=pos, select_node=False)
            node.node_config = module_spec.config
            node_names_to_ids[module_spec.name] = node.id
        # Setup node connections
        for module_spec in config.modules_specs:
            # Get node
            target_node: AbstractNode = self.get_node_by_id(node_names_to_ids[module_spec.name])
            # Iterate over inputs
            for port_name, port_map_list in module_spec.inputs.items():
                # Get port
                target_port: Port = target_node.get_input(port_name)
                # Connect port
                for port_map in port_map_list:
                    origin_node: AbstractNode = self.get_node_by_id(node_names_to_ids[port_map.origin])
                    origin_port: Port = origin_node.get_output('value' if isinstance(origin_node, SourceNode) else port_map.port)
                    origin_port.connect_to(target_port, push_undo=False, emit_signal=False)
            # Iterate over outputs
            for output_name, port_name in module_spec.outputs.items():
                # Get port
                origin_port: Port = target_node.get_output(port_name)
                # Get sink port
                sink_node: SinkNode = self.get_node_by_id(node_names_to_ids[output_name])
                sink_port: Port = sink_node.get_input('value')
                origin_port.connect_to(sink_port, push_undo=False, emit_signal=False)
        # Clean stack
        self.clear_selection()
        self.clear_undo_stack()
        self.center_on()
        self._is_dirty = False

    def validate_graph(self,) -> list[str]:
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

    def _gather_modules_specs(self, errors: list | None = None) -> list[ModuleSpecs]:
        """
            Build the controller modules spec list from the graph state.
        """
        modules_specs = []
        for node in self.all_nodes():
            if isinstance(node, (SourceNode, SinkNode)):
                continue
            else:
                modules_specs.append(node.get_module_spec())
        return modules_specs

    def _gather_controller_metadata(self, errors: list | None = None) -> dict:
        """
            Build the controller metadata from the graph state.
        """
        io_nodes_metadata = {}
        for node in self.all_nodes():
            if isinstance(node, (SourceNode, SinkNode)):
                io_nodes_metadata[node.NODE_NAME] = node.get_module_spec()
            else:
                continue
        return io_nodes_metadata

    def get_config_cls(self,) -> ControllerConfig:
        if self._controller_type == ControllerType.BRAIN:
            return BrainConfig
        elif self._controller_type == ControllerType.NEURON:
            return NeuronConfig
        else:
            raise ValueError(
                f'Unknown controller type "{self._controller_type}"'
            )

    def build_controller_config(self, is_partial: bool = True, errors: list | None = None) -> ControllerConfig:
        # Get controller class
        controller_cls = self.get_config_cls()
        # Gather modules and metadata
        modules_specs = self._gather_modules_specs()
        controller_metadata = self._gather_controller_metadata()
        # Construct brain config.
        if is_partial:
            brain_config = controller_cls._create_partial(
                modules_specs=modules_specs
            )
        else:
            if errors is not None:
                brain_config = controller_cls._create_partial(
                    modules_specs=modules_specs
                )
                brain_config.validate(errors=errors)
            else:
                brain_config = controller_cls(
                    modules_specs=modules_specs
                )
        brain_config.__graph_editor_metadata__ = controller_metadata
        return brain_config

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################