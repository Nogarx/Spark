#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.nn.brain import BrainConfig
    from spark.graph_editor.inspector import NodeInspectorWidget

import os
import sys
import json
import logging
import networkx as nx
import jax.numpy as jnp
import typing as tp
from Qt import QtWidgets, QtCore, QtGui
from NodeGraphQt import NodeGraph, Port
from spark.core.specs import ModuleSpecs, InputSpec, OutputSpec, PortMap
from spark.core.shape import Shape
from spark.nn.brain import BrainConfig
from spark.graph_editor.utils import MenuTree, JsonEncoder
from spark.graph_editor.nodes import module_to_nodegraph, SourceNode, SinkNode, AbstractNode, SparkModuleNode
#from spark.graph_editor.specs import InputSpecEditor, OutputSpecEditor, PortMap, ModuleSpecsEditor
from spark.graph_editor.graph import SparkNodeGraph
from spark.graph_editor.inspector import NodeInspectorWidget

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class SparkGraphEditor:

    # Singletons.
    app = None
    window = None
    graph = None
    inspector = None
    _current_session_path = None
    _current_model_path = None

    def __init__(self):
        # QApplication instance.
        if SparkGraphEditor.app is None:
            SparkGraphEditor.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        if SparkGraphEditor.window is None:
            SparkGraphEditor.window, SparkGraphEditor.inspector = self._window_init()

        # NodeGraph controller.
        if SparkGraphEditor.graph is None:
            SparkGraphEditor.graph = SparkNodeGraph()
            placeholder_widget = QtWidgets.QWidget()
            placeholder_widget.setStyleSheet("background-color: #333;")
            layout = QtWidgets.QVBoxLayout(placeholder_widget)
            layout.addWidget(SparkGraphEditor.graph.widget)
            # TODO: Fix bug: Widget zooms in when resizing inspector.
            SparkGraphEditor.window.setCentralWidget(placeholder_widget)
            # Menus
            context_menu = MenuTree(SparkGraphEditor.graph.get_context_menu('graph'))
            nodes_menu = SparkGraphEditor.graph.get_context_menu('nodes')

            # Register source/sink nodes.
            SparkGraphEditor.graph.register_node(SourceNode)
            context_menu['Interfaces'].add_command(
                SourceNode.NODE_NAME, 
                lambda *args, id='spark.SourceNode': self.create_node(*args, id=id)
            )
            SparkGraphEditor.graph.register_node(SinkNode)
            context_menu['Interfaces'].add_command(
                SinkNode.NODE_NAME, 
                lambda *args, id='spark.SinkNode': self.create_node(*args, id=id)
            )

            # Register nodes and add them to the context menu.
            from spark.core.registry import REGISTRY
            for key, entry in REGISTRY.MODULES.items():
                nodegraph_class = module_to_nodegraph(entry)
                SparkGraphEditor.graph.register_node(nodegraph_class)
                id = nodegraph_class.__identifier__ + f'._NG_{entry.class_ref.__name__}'
                context_menu[entry.path].add_command(
                    nodegraph_class.NODE_NAME, 
                    lambda *args, id=id: self.create_node(*args, id=id)
                )

            # Base commands
            context_menu.graph_menu_ref.add_separator()
            context_menu.add_command('Validate Topology', self.validate_graph)
            context_menu.add_command('Delete Selected', self.delete_selected, shortcut='del')

            # Base node commands
            nodes_menu.add_command('Inspect', self.delete_node, node_class=AbstractNode)
            #nodes_menu.add_separator()
            nodes_menu.add_command('Delete', self.delete_node, node_class=AbstractNode)

            #----- DEBUG NODES -----#.
            #----- DEBUG NODES -----#.
            source = SparkGraphEditor.graph.create_node('spark.SourceNode', name='input_0', pos=[0,300])
            sink = SparkGraphEditor.graph.create_node('spark.SinkNode', name='output_0', pos=[600,300])
            spiker = SparkGraphEditor.graph.create_node('spark._NG_PoissonSpiker', name='PoissonSpiker_0', pos=[150,0])
            neurons = SparkGraphEditor.graph.create_node('spark._NG_ALIFNeuron', name='ALIFNeuron_0', pos=[300,300])
            integrator = SparkGraphEditor.graph.create_node('spark._NG_ExponentialIntegrator', name='ExponentialIntegrator_0', pos=[450,0])
            source.set_output(0, spiker.input(0))
            spiker.set_output(0, neurons.input(0))
            neurons.set_output(0, integrator.input(0))
            neurons.set_output(0, neurons.input(0))
            integrator.set_output(0, sink.input(0))
            SparkGraphEditor.graph.clear_selection()
            SparkGraphEditor.graph.clear_undo_stack()
            SparkGraphEditor.graph.center_on()
            #----- DEBUG NODES -----#.
            #----- DEBUG NODES -----#.

            # Dynamic UI
            SparkGraphEditor.graph.node_selection_changed.connect(self._update_inspector_selection)
            SparkGraphEditor.graph.stateChanged.connect(self._update_ui_state)
            self._update_ui_state()

    def delete_node(self, graph: NodeGraph, node: AbstractNode) -> None:
        graph.delete_node(node)

    def delete_selected(self, graph: NodeGraph)  -> None:
        nodes: list[AbstractNode] = graph.selected_nodes()
        for node in nodes:
            graph.delete_node(node)

    def _validate_graph(self,) -> bool:
        errors = []
        nx_graph = SparkGraphEditor.graph.build_raw_graph()
        connected_components = list(nx.weakly_connected_components(nx_graph))
        # Check input/output exists.
        node_types = nx.get_node_attributes(nx_graph, "node_type").values()
        has_input = 'SourceNode' in node_types
        has_output = 'SinkNode' in node_types
        if not has_input:
            errors.append(f'No source node was found.')
        if not has_output:
            errors.append(f'No sink node was found.')
        # Check number of connected components.
        if len(connected_components) > 1:
            errors.append(f'Multiple connected components are not currently supported but found {len(connected_components)}.')
        return errors

    def validate_graph(self,) -> bool:
        errors = self._validate_graph()
        if len(errors) == 0: 
            QtWidgets.QMessageBox.information(
                SparkGraphEditor.graph.viewer(), 
                'Sucess!', f'No error was detected during validation.'
            )
        else: 
            QtWidgets.QMessageBox.warning(
                SparkGraphEditor.graph.viewer(), 
                'Invalid Model', 'The following errors were detected:\n\n  •  '+'\n\n  •  '.join(errors)+'\n'
            )

    def create_node(self, *args, id: str)  -> None:
        """
            Prompts the user for a node name using a dialog and then creates the node.
        """
        # TODO: Improve this dirty patch.
        base = id.split('.')[-1]
        it = 0
        if base == 'SourceNode':
            base = 'input'
        elif base == 'SinkNode':
            base = 'output'
        else:
            base = base.removeprefix('_NG_')
        current_name_attempt = f'{base}_{it}'
        while SparkGraphEditor.graph.name_exist(current_name_attempt):
            current_name_attempt = f'{base}_{it}'
            it +=1 
        while True:
            # Promt user for node's name
            name, ok = QtWidgets.QInputDialog.getText(
                SparkGraphEditor.graph.viewer(),
                "Create Node",
                f"Enter node's ID:",
                QtWidgets.QLineEdit.EchoMode.Normal,
                current_name_attempt)
            
            # Validate name
            if SparkGraphEditor.graph.name_exist(name):
                error_msg = f'The name "{name}" is already in use.'
                if not name:
                    error_msg = 'The name cannot be empty.'
                QtWidgets.QMessageBox.warning(SparkGraphEditor.graph.viewer(), 'Invalid Name', error_msg)
                current_name_attempt = name
            # Create the node if the user clicked "OK" and the name is not empty.
            elif ok and name:
                mouse_pos = SparkGraphEditor.graph.cursor_pos()
                SparkGraphEditor.graph.create_node(id, name=f'{name}', pos=mouse_pos, selected=True) 
                logging.info(f'Created node with ID "{id}" and name "{name}" at {mouse_pos}.')
                break
            # User closed the window
            elif not ok:
                break

    def launch(self) -> None:
        """
            Creates and shows the editor window without blocking.
            This method is safe to call multiple times.
        """
        if self.window.isVisible():
            self.window.activateWindow()
            self.window.raise_()
        else:
            self.window.show()

    def _maybe_save(self) -> bool:
        """
            Checks for unsaved changes and asks the user if they want to save.
            Returns True if the operation should proceed (user saved or discarded),
            and False if the operation should be cancelled.
        """
        if not SparkGraphEditor.graph._is_modified:
            return True
        
        msg_box = QtWidgets.QMessageBox(None)
        msg_box.setText("The document has been modified.")
        msg_box.setInformativeText("Do you want to save your changes?")
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Save |
            QtWidgets.QMessageBox.StandardButton.Discard |
            QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Save)
        
        ret = msg_box.exec()
        
        if ret == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save_session()
        elif ret == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        return True

    def new_session(self) -> None:
        """
            Clears the current session after checking for unsaved changes.
        """
        if self._maybe_save():
            self.graph.clear_session()
            self._current_session_path = None
            SparkGraphEditor.graph._is_modified = False
            self._update_ui_state()

    def save_session(self) -> bool:
        """
            Saves the current session to a Spark Graph Editor file.
        """
        if self._current_session_path is None:
            return self.save_session_as()
        else:
            try:
                self.graph.save_session(self._current_session_path)
                SparkGraphEditor.graph._is_modified = False
                self._update_ui_state()
                return True
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not save file:\n{e}")
                return False

    def save_session_as(self) -> bool:
        """
            Saves the current session to a new Spark Graph Editor file.
        """
        dialog = QtWidgets.QFileDialog(None, 'Save Session As')
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter('Spark Graph Files (*.sge);;All Files (*)')
        dialog.setDefaultSuffix('sge')
        while dialog.exec():
            path = dialog.selectedFiles()[0]
            if os.path.exists(path):
                ret = QtWidgets.QMessageBox.question(
                    None,
                    'Confirm Overwrite',
                    f'The file "{os.path.basename(path)}" already exists.<br>Do you want to replace it?',
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No
                )
                if ret == QtWidgets.QMessageBox.StandardButton.No:
                    continue 
            self._current_session_path = path
            return self.save_session()
        
        return False

    def load_session(self) -> None:
        """
            Loads a graph state from a Spark Graph Editor file after checking for unsaved changes.
        """
        if not self._maybe_save():
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
                        parent=None, 
                        caption='Load Session', 
                        filter='Spark Graph Files (*.sge);;All Files (*)'
        )
        if path:
            try:
                self.graph.load_session(path)
                self._current_session_path = path
                SparkGraphEditor.graph._is_modified = False
                self._update_ui_state()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not load file:\n{e}")

    # TODO: This function will not work without some extra metada inside the brain config or 
    # some cleaver way to resolve node placemenet. The second option is prefered.
    def load_from_model(self) -> None:
        """
            Loads a graph state from a Spark configuration file after checking for unsaved changes.
        """
        if not self._maybe_save():
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
                        parent=None, 
                        caption='Load model', 
                        filter='Spark Cfg Files (*.scfg);;All Files (*)'
        )
        if path:
            try:
                # TODO: Add nodes dynamically
                #self.graph.load_session(path)
                self._current_session_path = path
                SparkGraphEditor.graph._is_modified = False
                self._update_ui_state()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not load file:\n{e}")


    def export_model(self) -> bool:
        """
            Exports the graph state to a Spark configuration file.
        """
        if self._current_model_path is None:
            return self.export_model_as()
        else:
            #try:
                # Validate model.
                errors = self._validate_graph()
                if len(errors) > 0: 
                    QtWidgets.QMessageBox.warning(
                        SparkGraphEditor.graph.viewer(), 
                        'Invalid Model', 'The following errors were detected:\n\n  •  '+'\n\n  •  '.join(errors)+'\n'
                    )
                    return 
                
                brain_config = self._build_brain_config()
                brain_config.to_file(self._current_model_path)
                return True
            #except Exception as e:
            #    QtWidgets.QMessageBox.critical(None, 'Error', f'Could not save file:\n{e}')
            #    return False

    def export_model_as(self) -> bool:
        """
            Exports the graph state to a new Spark configuration file.
        """
        dialog = QtWidgets.QFileDialog(None, 'Save Session As')
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter('Spark Cfg File (*.scfg);;All Files (*)')
        dialog.setDefaultSuffix('scfg')

        while dialog.exec():
            path = dialog.selectedFiles()[0]
            if os.path.exists(path):
                ret = QtWidgets.QMessageBox.question(
                    None,
                    'Confirm Overwrite',
                    f'The file "{os.path.basename(path)}" already exists.<br>Do you want to replace it?',
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No
                )
                if ret == QtWidgets.QMessageBox.StandardButton.No:
                    continue 
            self._current_model_path = path
            return self.export_model()
        
        return False

    def _build_brain_config(self,) -> BrainConfig:
        """
            Build the model from the graph state.
        """
        input_map: dict[str, InputSpec] = {}
        output_map: dict[str, OutputSpec] = {}
        modules_map: dict[str, ModuleSpecs] = {}
        for node in SparkGraphEditor.graph.all_nodes():
            if isinstance(node, SourceNode):
                # Source nodes had a single output of the appropiate type
                input_map[node.NODE_NAME] = InputSpec(
                    payload_type=node.output_specs['value'].payload_type,
                    shape=node.output_specs['value'].shape if node.output_specs['value'].shape else Shape(1,),
                    dtype=node.output_specs['value'].dtype if node.output_specs['value'].dtype else jnp.float16,
                    is_optional=False,
                    description=node.output_specs['value'].description,
                )
            elif isinstance(node, SinkNode):
                # Sink nodes are virtual nodes, we need to get the origin of its input to resolve the output_map entry. 
                # NOTE: Sink nodes are only allowed a single input so its safe to resolve it this way.
                input_port: Port = node.input_ports()[0]
                origin_port: Port = input_port.connected_ports()[0]
                origin_node: AbstractNode = origin_port.node()
                if isinstance(origin_node, SourceNode):
                    origin_name = '__call__'
                    port_name = origin_node.NODE_NAME
                else:
                    origin_name = origin_node.NODE_NAME
                    port_name = origin_port.name()
                #origin_node: AbstractNode = SparkGraphEditor.graph.get_node_by_name(
                #    output_map[node.NODE_NAME]['port_maps'][0]['origin']
                #)
                # Register node in output_map if not present
                if not (origin_name in output_map):
                     output_map[origin_name] = {}
                # Create OutputSpec
                output_map[origin_name][port_name] = OutputSpec(
                    payload_type=origin_node.output_specs[port_name].payload_type,
                    shape=origin_node.output_specs[port_name].shape if origin_node.output_specs[port_name].shape else Shape(1,),
                    dtype=origin_node.output_specs[port_name].dtype if origin_node.output_specs[port_name].dtype else jnp.float16,
                    description=origin_node.output_specs[port_name].description,
                )
            elif isinstance(node, SparkModuleNode):
                # Gather input maps
                inputs = {}
                for key, input_spec in node.input_specs.items():
                    port_maps = []
                    for p_m in input_spec.port_maps:
                        # Tranform origin id to name
                        origin_node: AbstractNode = SparkGraphEditor.graph.get_node_by_id(p_m.origin)
                        if isinstance(origin_node, SourceNode):
                            origin_name = '__call__'
                            port_name = origin_node.NODE_NAME
                        else:
                            origin_name = origin_node.NODE_NAME
                            port_name = p_m.port
                        port_maps.append(PortMap(origin=origin_name, port=port_name))
                    inputs[key] = port_maps
                # Build module spec
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
        return brain_config


    def closeEvent(self, event)-> None:
        """
            Overrides the default close event to check for unsaved changes.
        """
        if self._maybe_save():
            event.accept()
        else:
            event.ignore()

    def _window_init(self,) -> tuple[QtWidgets.QMainWindow, NodeInspectorWidget]:
        # Create base window.
        window = QtWidgets.QMainWindow()
        window.setWindowTitle('Spark Graph Editor')
        window.resize(1366, 768)

        # --- Toolbar Setup ---
        menu_bar = window.menuBar()
        file_menu = menu_bar.addMenu('File')
        
        # Add actions to the menu.
        new_action = file_menu.addAction('New')
        new_action.triggered.connect(self.new_session)
        file_menu.addSeparator()
        load_action = file_menu.addAction('Load')
        load_action.triggered.connect(self.load_session)
        #load_from_model_action = file_menu.addAction('Load from model')
        #load_from_model_action.triggered.connect(self.load_from_model)
        file_menu.addSeparator()
        save_action = file_menu.addAction('Save')
        save_action.triggered.connect(self.save_session)
        save_as_action = file_menu.addAction('Save as ...')
        save_as_action.triggered.connect(self.save_session)
        file_menu.addSeparator()
        export_action = file_menu.addAction('Export model')
        export_action.triggered.connect(self.export_model)
        export_as_action = file_menu.addAction('Export model as...')
        export_as_action.triggered.connect(self.export_model_as)
        file_menu.addSeparator()
        exit_action = file_menu.addAction('Exit')

        # Get references to dynamic actions
        self.save_action = save_action
        self.export_action = export_action

        # --- Edit Menu ---
        # Add the second menu. The menu bar will handle the interaction between them.
        edit_menu = menu_bar.addMenu('Edit')
        undo_action = edit_menu.addAction('Undo')
        redo_action = edit_menu.addAction('Redo')

        # Add Bottom Toolbar
        bottom_toolbar = QtWidgets.QToolBar('Bottom Toolbar', window, movable=False)
        window.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, bottom_toolbar)
        bottom_toolbar.addWidget(QtWidgets.QLabel('Status: Ready'))

        # Add inspection dock
        inspection_dock = QtWidgets.QDockWidget('', window,)
        inspection_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        inspection_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        inspector = NodeInspectorWidget()
        inspection_dock.setWidget(inspector)
        window.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, inspection_dock)

        # Default inspector size.
        window.resizeDocks([inspection_dock], [1], QtCore.Qt.Orientation.Vertical)
        return window, inspector

    def _update_inspector_selection(self, new_selection: list[AbstractNode], previous_selection: list[AbstractNode]) -> None:
        if len(new_selection) == 1:
            SparkGraphEditor.inspector.set_node(new_selection[0])
        else: 
            SparkGraphEditor.inspector.set_node(None)

    def _update_ui_state(self) -> None:
        """
            Updates all state-dependent UI elements like titles and action labels.
        """
        # Update window title to show file name and modification status
        base_title = 'Spark Graph Editor'
        if self._current_session_path:
            file_name = os.path.basename(self._current_session_path)
            modified_marker = ' *' if SparkGraphEditor.graph._is_modified else ''
            SparkGraphEditor.window.setWindowTitle(f'{base_title} - Untitled{modified_marker}')
        else:
            modified_marker = ' *' if SparkGraphEditor.graph._is_modified else ''
            SparkGraphEditor.window.setWindowTitle(f'{base_title} - Untitled{modified_marker}')
        
        # Update 'Save' action text and enabled state
        self.save_action.setEnabled(SparkGraphEditor.graph._is_modified)
        
        # Update 'Export' action text
        if self._current_model_path:
            self.export_action.setText(f"Export '{os.path.basename(self._current_model_path)}'...")
        else:
            self.export_action.setText("Export Model...")

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################