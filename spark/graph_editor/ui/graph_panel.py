#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
    
import typing as tp
from PySide6 import QtCore, QtWidgets, QtGui
import PySide6QtAds as ads

from spark.core.registry import REGISTRY
from spark.nn.controllers.base import Controller
from spark.graph_editor.models.graph import SparkNodeGraph, ControllerType
from spark.graph_editor.models.graph_menu_tree import HierarchicalMenuTree
from spark.graph_editor.models.nodes import SourceNode, SinkNode, AbstractNode, module_to_nodegraph
from spark.graph_editor.ui.console_panel import MessageLevel

# NOTE: Small workaround to at least have base autocompletion.
if tp.TYPE_CHECKING:
    CDockWidget = QtWidgets.QWidget
else:
    CDockWidget = ads.CDockWidget

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class GraphPanel(CDockWidget):
    """
        Container for the NodeGraphQt object.
    """

    broadcast_message = QtCore.Signal(MessageLevel, str)
    onWidgetUpdate = QtCore.Signal(str, tp.Any)

    def __init__(self, controller_type: ControllerType, **kwargs):
        super().__init__('Graph', **kwargs)
        # Initialize the graph controller.
        self.graph = SparkNodeGraph(controller_type=controller_type)
        # Add graph widget to layout.
        self.layout().addWidget(self.graph.widget)
        # Setup graph context menu.
        self._context_menu = self._setup_context_menu()
        self._current_controller_type = None
        self.on_controller_type_change(controller_type)

    def _setup_context_menu(self,) -> HierarchicalMenuTree:
        """
            Setup NodeGraphQt node classes using a Spark.Module to Node factory (module_to_nodegraph).
        """

        # Menus
        context_menu = HierarchicalMenuTree(self.graph.get_context_menu('graph'))
        
        # Add source/sink node models.
        cls_name = SourceNode.__name__
        context_menu.add_command(
            cls_name, 
            lambda *args, cls=cls_name: self.maybe_create_node(*args, nodegraph_cls=cls)
        )
        cls_name = SinkNode.__name__
        context_menu.add_command(
            cls_name, 
            lambda *args, cls=cls_name: self.maybe_create_node(*args, nodegraph_cls=cls)
        )
        context_menu.graph_menu_ref.add_separator()
        # Add module node models.
        for key, entry in REGISTRY.MODULES.items():
            cls_name = entry.class_ref.__name__
            if issubclass(entry.class_ref, (Controller)):
                continue
            context_menu[entry.path].add_command(
                cls_name, 
                lambda *args, cls=cls_name: self.maybe_create_node(*args, nodegraph_cls=cls)
            )
        # Add neuron node models.
        for key, entry in REGISTRY.NEURONS.items():
            cls_name = entry.class_ref.__name__
            context_menu[entry.path].add_command(
                cls_name, 
                lambda *args, cls=cls_name: self.maybe_create_neuron(*args, nodegraph_cls=cls)
            )
        # Base commands
        context_menu.graph_menu_ref.add_separator()
        #context_menu.add_command('Validate Topology', self.validate_graph)
        context_menu.add_command('Delete Selected', self.delete_selected, shortcut='del')
        return context_menu
        
    def maybe_create_neuron(self, *args, nodegraph_cls: str) -> None:
        if self._current_controller_type == ControllerType.BRAIN:
            self.maybe_create_node(*args, nodegraph_cls=nodegraph_cls)
        else:
            entry = REGISTRY.NEURONS.get(nodegraph_cls)
            config = entry.class_ref.get_config_spec()()
            self.graph.load_neuron_from_model(config)
            #self._context_menu['Neurons'].graph_menu_ref.qmenu.setEnabled(False)

    def on_controller_type_change(self, controller_type: ControllerType):
        self._current_controller_type = controller_type
        #if controller_type == ControllerType.BRAIN:
        #    self._context_menu['Neurons'].graph_menu_ref.qmenu.setEnabled(True)
        #else:
        #    self._context_menu['Neurons'].graph_menu_ref.qmenu.setEnabled(False)

    def delete_selected(self,) -> None:
        try:
            nodes = self.graph.selected_nodes()
            nodes_names = [n.NODE_NAME for n in nodes]
            self.graph.delete_nodes(nodes)
            for n in nodes_names:
                self.broadcast_message.emit(MessageLevel.INFO, f'Node {n} deleted sucessfully.')
        except Exception as e:
            self.broadcast_message.emit(MessageLevel.ERROR, f'Encounter an error while trying to delete nodes: {e}')

    def maybe_create_node(self, *args, nodegraph_cls: str)  -> None:
        """
            Prompts the user for a node name using a dialog and then creates the node.

            Input:
                id: str, node class id of the node to create.
        """

        # Generate a generic name for the new node.
        base_name = f'{nodegraph_cls}'
        it = 0
        current_name_attempt = f'{base_name}_{it}'
        while self.graph.name_exist(current_name_attempt):
            current_name_attempt = f'{base_name}_{it}'
            it +=1 

        # Start user interaction.
        while True:

            # Promt user for node's name.
            name, ok = QtWidgets.QInputDialog.getText(
                self.graph.viewer(),
                'Create Node',
                f'Enter node\'s ID:',
                QtWidgets.QLineEdit.EchoMode.Normal,
                current_name_attempt
            )

            # Validate name
            if self.graph.name_exist(name):
                error_msg = f'The name \"{name}\" is already in use.'
                if not name:
                    error_msg = 'The name cannot be empty.'
                QtWidgets.QMessageBox.warning(self.graph.viewer(), 'Invalid Name', error_msg)
                current_name_attempt = name

            # Create the node if the user clicked OK and the name is not empty.
            elif ok and name:
                mouse_pos = self.graph.cursor_pos()
                node_type = f'spark.{nodegraph_cls}'
                self.graph.create_node(node_type, name=f'{name}', pos=mouse_pos, selected=True) 
                break

            # User closed the window
            elif not ok:
                break

    def _debug_model(self,) -> None:
        #----- DEBUG NODES -----#.
        #----- DEBUG NODES -----#.
        source = self.graph.create_node('spark.SourceNode', name='input_0', pos=[0,300])
        sink = self.graph.create_node('spark.SinkNode', name='output_0', pos=[600,300])
        spiker = self.graph.create_node('spark.PoissonSpiker', name='PoissonSpiker_0', pos=[150,0])
        neurons = self.graph.create_node('spark.ALIFNeuron', name='ALIFNeuron_0', pos=[300,300])
        integrator = self.graph.create_node('spark.ExponentialIntegrator', name='ExponentialIntegrator_0', pos=[450,0])
        source.set_output(0, spiker.input(0))
        spiker.set_output(0, neurons.input(0))
        neurons.set_output(0, integrator.input(0))
        neurons.set_output(0, neurons.input(0))
        integrator.set_output(0, sink.input(0))
        self.graph.clear_selection()
        self.graph.clear_undo_stack()
        self.graph.center_on()
        #----- DEBUG NODES -----#.
        #----- DEBUG NODES -----#.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################