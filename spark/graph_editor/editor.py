#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import sys
import json
import logging

from shiboken6 import isValid
from PySide6.QtWidgets import QApplication, QMainWindow, QDockWidget, QStatusBar, QLabel, QFileDialog
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, Signal

from spark.graph_editor.view.node_item import NodeItem
from spark.graph_editor.view.graph_view import GraphScene, GraphView
from spark.graph_editor.widgets.hierarchy_view import HierarchyView
from spark.graph_editor.widgets.inspector_view import InspectorView
from spark.graph_editor.widgets.console_view import ConsoleView, MessageLevel
from spark.graph_editor.widgets.preferences_dialog import PreferencesDialog
from spark.graph_editor.styles.manager import STYLES

#from debug import register_debug_port_types, populate_debug_graph

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class SparkGraphEditor:

    def __init__(self) -> None:

        # Check if the editor was launched in the CLI or is using IPykernel
        self._is_interactive = 'ipykernel' in sys.modules
        if self._is_interactive:
            # Integrate Qt event loop to avoid the %gui qt in the terminal
            from IPython import get_ipython
            get_ipython().enable_gui('qt')

        # QApplication instance.
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        self.window: GraphEditorWindow | None = None

    def launch(self) -> None:
        """
            Creates and shows the editor window without blocking.
        """
        # If a previous window exists, explicitly delete it (safe)
        if self.window is not None:
            self.window.close()
            self.window.deleteLater()
            del self.window

        # Set app style
        STYLES.init()
        STYLES.reloaded.connect(lambda app=self.app: STYLES.apply(app))
        STYLES.apply(self.app)

        # Ask for controller type
        #controller_type = self.set_session_controller_type(True)
        #if controller_type is not None:self._scene
        # Create base window.
        self.window = GraphEditorWindow()
        self.window.windowClosed.connect(self.exit_editor)
        
        # DEBUG: SECOND SCREEN
        screens = self.app.screens()
        if len(screens) > 1:
            target_screen = screens[1]
            target_geo = target_screen.availableGeometry()
            self.window.setScreen(target_screen)
            self.window.setGeometry(target_geo)

        self.window.showMaximized()

        # Start loop
        self.app.exec_()
    
    def exit_editor(self,) -> None:
        """
            Exit editor.
        """
        self.app.quit()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class GraphEditorWindow(QMainWindow):

    windowClosed = Signal() 

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Spark Graph Editor')
        self.setDockNestingEnabled(True)
        # Central Canvas (Graph Editor)
        self._scene = GraphScene()
        self.view = GraphView(self._scene)
        self.setCentralWidget(self.view)
        self.current_file = None
        # Setup Status Bar
        self._statusBar = QStatusBar()
        self._statusBar.setObjectName('statusBar') 
        self.setStatusBar(self._statusBar)
        self._statusBar.showMessage('Ready')
        # Setup Docks
        self._setup_docks()
        # Connect selection changes to Inspector
        self._scene.selectionChanged.connect(self._on_selection_changed)
        # Connect double-click on hierarchy to canvas centering
        self.dock_hierarchy.widget().node_double_clicked.connect(self._center_on_node)
        # Setup menus (must be after docks so they can be toggled)
        self._setup_menus()
        self.reload_ui()
        logger = logging.getLogger('spark')
        logger.info('Editor Initialized...')


    def closeEvent(self, event) -> None:
        #self.save_layout()
        super().closeEvent(event)
        self.windowClosed.emit()

    def _center_on_node(self, node_model) -> None:
        if not node_model: return
        for item in self._scene.items():
            if isinstance(item, NodeItem) and item.model == node_model:
                self.view.centerOn(item)
                break

    def _on_selection_changed(self) -> None:
        if not isValid(self._scene): 
            return
        selected_nodes = [item.model for item in self._scene.selectedItems() if isinstance(item, NodeItem)]
        # If exactly one node is selected, show it in the inspector
        if len(selected_nodes) == 1:
            self.dock_inspector.widget().set_node(selected_nodes[0], self._scene.model)
        else:
            self.dock_inspector.widget().set_node(None)

    def _create_dock_title(self, title) -> QLabel:
        label = QLabel(f' {title.upper()}')
        label.setObjectName('dockTitle')
        return label

    def _setup_docks(self) -> None:
        # Set corners so left and right docks extend to the bottom, sandwiching the console
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        # Left: Hierarchy
        self.dock_hierarchy = QDockWidget('Hierarchy', self)
        self.dock_hierarchy.setTitleBarWidget(self._create_dock_title('Hierarchy'))
        self.dock_hierarchy.setWidget(HierarchyView(self._scene.model))
        self.dock_hierarchy.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_hierarchy)
        # Right: Inspector
        self.dock_inspector = QDockWidget('Inspector', self)
        self.dock_inspector.setTitleBarWidget(self._create_dock_title('Inspector'))
        self.dock_inspector.setWidget(InspectorView())
        self.dock_inspector.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_inspector)
        # Bottom: Console
        self.dock_console = QDockWidget('Console', self)
        self.dock_console.setTitleBarWidget(self._create_dock_title('Console'))
        self.dock_console.setWidget(ConsoleView())
        self.dock_console.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_console)

    def _setup_menus(self) -> None:
        menubar = self.menuBar()
        # File Menu
        file_menu = menubar.addMenu('&File')
        new_action = QAction('New Session', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.new_graph)
        file_menu.addAction(new_action)
        load_action = QAction('Load Session...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_graph)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        save_action = QAction('Save Session', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_graph)
        file_menu.addAction(save_action)
        save_as_action = QAction('Save Session As...', self)
        save_as_action.setShortcut('Ctrl+Shift+S')
        save_as_action.triggered.connect(self.save_graph_as)
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        quit_action = QAction('Quit', self)
        quit_action.setShortcut('Ctrl+Q')
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        # Edit Menu
        edit_menu = menubar.addMenu('&Edit')
        undo_action = self._scene.model.undo_stack.createUndoAction(self, '&Undo')
        undo_action.setShortcut('Ctrl+Z')
        edit_menu.addAction(undo_action)
        redo_action = self._scene.model.undo_stack.createRedoAction(self, '&Redo')
        redo_action.setShortcut('Ctrl+Shift+Z')
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()
        prefs_action = QAction('Preferences...', self)
        prefs_action.triggered.connect(self.open_preferences)
        edit_menu.addAction(prefs_action)
        # Window Menu
        window_menu = menubar.addMenu('&Window')
        hierarchy_action = self.dock_hierarchy.toggleViewAction()
        hierarchy_action.setText('Hierarchy')
        window_menu.addAction(hierarchy_action)
        inspector_action = self.dock_inspector.toggleViewAction()
        inspector_action.setText('Inspector')
        window_menu.addAction(inspector_action)
        console_action = self.dock_console.toggleViewAction()
        console_action.setText('Console')
        window_menu.addAction(console_action)

    def new_graph(self) -> None:
        self._scene.model.clear()
        self.current_file = None
        self._statusBar.showMessage('New session started.')
        logging.getLogger('spark').info('Started a new empty session.')

    def save_graph(self) -> None:
        if not self.current_file:
            self.save_graph_as()
            return
        self._save_to_file(self.current_file)

    def save_graph_as(self) -> None:
        # Passing None as parent detaches the dialog from the main window's Qt style tree,
        # forcing the OS native dialog to be used instead.
        file_name, _ = QFileDialog.getSaveFileName(None, 'Save Session As', '', 'JSON Files (*.json);;All Files (*)')
        if file_name:
            self.current_file = file_name
            self._save_to_file(file_name)

    def _save_to_file(self, file_path) -> None:
        try:
            data = self._scene.model.to_dict()
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            self._statusBar.showMessage(f'Session saved to {file_path}')
            logging.getLogger('spark').log(MessageLevel.SUCCESS.value, f'Graph saved successfully to {file_path}')
        except Exception as e:
            self._statusBar.showMessage(f'Error saving session: {e}')
            logging.getLogger('spark').error(f'Error saving graph: {e}')

    def load_graph(self) -> None:
        # Passing None as parent forces the OS native dialog.
        file_name, _ = QFileDialog.getOpenFileName(None, 'Load Session', '', 'JSON Files (*.json);;All Files (*)')
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    data = json.load(f)
                self._scene.model.from_dict(data)
                self.current_file = file_name
                self._statusBar.showMessage(f'Session loaded from {file_name}')
                logging.getLogger('spark').log(MessageLevel.SUCCESS.value, f'Graph loaded successfully from {file_name}')
            except Exception as e:
                self._statusBar.showMessage(f'Error loading session: {e}')
                logging.getLogger('spark').error(f'Error loading graph: {e}')

    def open_preferences(self) -> None:
        dialog = PreferencesDialog(self)
        if dialog.exec():
            self.reload_ui()

    def reload_ui(self) -> None:
        # Save state
        model = self._scene.model
        current_file = self.current_file
        selected_nodes = [item.model for item in self._scene.selectedItems() if isinstance(item, NodeItem)]
        # Recreate Scene & View
        new_scene = GraphScene(model)
        new_view = GraphView(new_scene)
        self.setCentralWidget(new_view)
        self._scene.deleteLater()
        self.view.deleteLater()
        self._scene = new_scene
        self.view = new_view
        # Recreate Hierarchy
        new_hierarchy = HierarchyView(model)
        self.dock_hierarchy.setWidget(new_hierarchy)
        # Recreate Inspector
        new_inspector = InspectorView()
        self.dock_inspector.setWidget(new_inspector)
        # Recreate Console
        new_console = ConsoleView()
        self.dock_console.setWidget(new_console)
        # Reconnect
        self._scene.selectionChanged.connect(self._on_selection_changed)
        self.dock_hierarchy.widget().node_double_clicked.connect(self._center_on_node)
        # Restore selection visually
        for item in self._scene.items():
            if isinstance(item, NodeItem) and item.model in selected_nodes:
                item.setSelected(True)
        self._on_selection_changed()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################