#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import sys
import json
import pathlib
import logging

from shiboken6 import isValid
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QStatusBar, QLabel, QFileDialog, QStackedWidget, QMessageBox
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, Signal

from spark.graph_editor.view.node_item import NodeItem
from spark.graph_editor.view.graph_view import GraphScene, GraphView
from spark.graph_editor.widgets.hierarchy_view import HierarchyView
from spark.graph_editor.widgets.inspector_view import InspectorView
from spark.graph_editor.widgets.console_view import ConsoleView, MessageLevel
from spark.graph_editor.widgets.preferences_dialog import PreferencesDialog
from spark.graph_editor.widgets.controller_selection import StartView, NewModelDialog
from spark.graph_editor.models.controller_profile import ControllerProfile, profile_for_config
from spark.graph_editor.models import session_io
from spark.graph_editor.styles.manager import STYLES
from spark.graph_editor.styles import resources as icons

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
        # NOTE: The editor starts without a document. The canvas only becomes available once the user picks
        # the controller to build, since the controller dictates the palette and the exported configuration.
        self._scene = GraphScene()
        self.view = GraphView(self._scene)
        self._start_view = StartView()
        self._start_view.model_requested.connect(self.new_graph)
        self._start_view.open_requested.connect(self.load_session)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._start_view)
        self._stack.addWidget(self.view)
        self.setCentralWidget(self._stack)
        # A document has two paths: the session it is edited in and the model it exports to.
        self._session_path: pathlib.Path | None = None
        self._model_path: pathlib.Path | None = None
        # Setup Status Bar
        self._statusBar = QStatusBar()
        self._statusBar.setObjectName('statusBar')
        self.setStatusBar(self._statusBar)
        self._statusBar.showMessage('Ready')
        # Permanent indicator of the controller being built.
        self._controller_icon = QLabel()
        self._controller_icon.setObjectName('statusControllerIcon')
        self._statusBar.addPermanentWidget(self._controller_icon)
        self._controller_label = QLabel('')
        self._controller_label.setObjectName('statusControllerLabel')
        self._statusBar.addPermanentWidget(self._controller_label)
        # Setup Docks
        self._setup_docks()
        # Connect selection changes to Inspector
        self._scene.selectionChanged.connect(self._on_selection_changed)
        # Connect double-click on hierarchy to canvas centering
        self.dock_hierarchy.widget().node_double_clicked.connect(self._center_on_node)
        # Setup menus (must be after docks so they can be toggled)
        self._setup_menus()
        self._scene.model.profile_changed.connect(self._on_profile_changed)
        self.reload_ui()
        self._update_document_state()
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
        # NOTE: Menus are kept as attributes so they stay addressable from code without going through the
        # menu bar (reaching them through temporaries invalidates the PySide wrapper).
        # File Menu
        self._file_menu = file_menu = menubar.addMenu('&File')
        new_action = QAction('New Session', self)
        new_action.setShortcut('Ctrl+N')
        # NOTE: QAction.triggered carries the checked state, which must not be mistaken for a profile.
        new_action.triggered.connect(lambda _checked=False: self.new_graph())
        file_menu.addAction(new_action)
        load_action = QAction('Load Session...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(lambda _checked=False: self.load_session())
        file_menu.addAction(load_action)
        save_action = QAction('Save Session', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(lambda _checked=False: self.save_session())
        file_menu.addAction(save_action)
        save_as_action = QAction('Save Session As...', self)
        save_as_action.setShortcut('Ctrl+Shift+S')
        save_as_action.triggered.connect(lambda _checked=False: self.save_session_as())
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        # NOTE: A session is work in progress and always saves. A model is a finished controller and only
        # exports once the graph describes something the framework can instantiate.
        export_action = QAction('Export Model', self)
        export_action.triggered.connect(lambda _checked=False: self.export_model())
        file_menu.addAction(export_action)
        export_as_action = QAction('Export Model As...', self)
        export_as_action.triggered.connect(lambda _checked=False: self.export_model_as())
        file_menu.addAction(export_as_action)
        import_action = QAction('Import Model...', self)
        import_action.triggered.connect(lambda _checked=False: self.import_model_file())
        file_menu.addAction(import_action)
        file_menu.addSeparator()
        # Actions that require an open document.
        self._document_actions = [save_action, save_as_action, export_action, export_as_action, import_action]
        quit_action = QAction('Quit', self)
        quit_action.setShortcut('Ctrl+Q')
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        # Edit Menu
        self._edit_menu = edit_menu = menubar.addMenu('&Edit')
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
        self._window_menu = window_menu = menubar.addMenu('&Window')
        hierarchy_action = self.dock_hierarchy.toggleViewAction()
        hierarchy_action.setText('Hierarchy')
        window_menu.addAction(hierarchy_action)
        inspector_action = self.dock_inspector.toggleViewAction()
        inspector_action.setText('Inspector')
        window_menu.addAction(inspector_action)
        console_action = self.dock_console.toggleViewAction()
        console_action.setText('Console')
        window_menu.addAction(console_action)

    def new_graph(self, profile: ControllerProfile | None = None) -> None:
        """
            Starts a new model. The controller is asked for whenever it was not provided.
        """
        # Guard against signals that carry an unrelated payload (e.g. QAction.triggered(bool)).
        if not isinstance(profile, ControllerProfile):
            profile = None
        if profile is None:
            dialog = NewModelDialog(self)
            if not dialog.exec():
                return
            profile = dialog.selected_profile
        if not isinstance(profile, ControllerProfile):
            return
        self._scene.model.clear()
        self._scene.model.set_profile(profile, force=True)
        self._session_path = None
        self._model_path = None
        self._update_document_state()
        self._statusBar.showMessage(f'New {profile.label.lower()} session started.')
        logging.getLogger('spark').info(f'Started a new empty {profile.label} session.')

    def _on_profile_changed(self, profile: ControllerProfile | None) -> None:
        self._update_document_state()

    def _update_document_state(self) -> None:
        """
            Synchronizes the window with the presence (and type) of a document.
        """
        profile = self._scene.model.profile
        has_document = profile is not None
        # Canvas or start screen.
        self._stack.setCurrentWidget(self.view if has_document else self._start_view)
        # Title.
        file_name = self._session_path.stem if self._session_path else 'Untitled'
        if has_document:
            self.setWindowTitle(f'Spark Graph Editor - {profile.label} - {file_name}')
            icon_size = STYLES.get_val('start', 'status_icon_size', default=14)
            self._controller_icon.setPixmap(icons.get_pixmap(profile.icon, icon_size))
            self._controller_label.setText(profile.label)
        else:
            self.setWindowTitle('Spark Graph Editor')
            self._controller_icon.clear()
            self._controller_label.setText('')
        # Document dependent actions.
        for action in getattr(self, '_document_actions', []):
            action.setEnabled(has_document)
        if hasattr(self, '_controller_action'):
            self._controller_action.setEnabled(has_document and self._scene.model.can_change_profile())

    #-------------------------------------------------------------------------------------------------------#
    # Sessions and models
    #-------------------------------------------------------------------------------------------------------#

    # NOTE: A session (.sge) is the document being edited: it always saves, however incomplete it is, and it
    # remembers where the nodes are. A model (.scfg) is the finished controller handed to the framework: it
    # only exports when the graph describes something that can actually be instantiated.

    def save_session(self) -> bool:
        if self._scene.model.profile is None:
            return False
        if self._session_path is None:
            return self.save_session_as()
        return self._write_session(self._session_path)

    def save_session_as(self) -> bool:
        if self._scene.model.profile is None:
            return False
        # Passing None as parent detaches the dialog from the main window's Qt style tree,
        # forcing the OS native dialog to be used instead.
        file_name, _ = QFileDialog.getSaveFileName(None, 'Save Session As', '', session_io.SESSION_FILTER)
        if not file_name:
            return False
        return self._write_session(pathlib.Path(file_name))

    def _write_session(self, path: pathlib.Path) -> bool:
        try:
            written = session_io.save_session(self._scene.model, path)
        except Exception as error:
            self._report_error('Unable to save the session', str(error))
            return False
        self._session_path = written
        self._update_document_state()
        self._statusBar.showMessage(f'Session saved to {written}')
        logging.getLogger('spark').log(MessageLevel.SUCCESS.value, f'Session saved to "{written}".')
        return True

    def load_session(self) -> None:
        # Passing None as parent forces the OS native dialog.
        file_name, _ = QFileDialog.getOpenFileName(None, 'Load Session', '', session_io.SESSION_FILTER)
        if not file_name:
            return
        try:
            session = session_io.load_session(file_name)
        except Exception as error:
            self._report_error('Unable to load the session', str(error))
            return
        if session.profile is None:
            self._report_error('Unable to load the session', 'The controller of this session is not registered.')
            return
        model = self._scene.model
        model.clear()
        # The controller comes from the file, the user is never asked when loading.
        model.set_profile(session.profile, force=True)
        model.controller_config = session.config
        self.view.import_config(session.config, label=pathlib.Path(file_name).stem, layout=session.layout)
        model.undo_stack.clear()
        self._session_path = pathlib.Path(file_name)
        self._model_path = None
        self._update_document_state()
        self._statusBar.showMessage(f'Session loaded from {file_name}')
        logging.getLogger('spark').log(MessageLevel.SUCCESS.value, f'Session loaded from "{file_name}".')

    def export_model(self) -> bool:
        if self._scene.model.profile is None:
            return False
        if self._model_path is None:
            return self.export_model_as()
        return self._write_model(self._model_path)

    def export_model_as(self) -> bool:
        if self._scene.model.profile is None:
            return False
        file_name, _ = QFileDialog.getSaveFileName(None, 'Export Model As', '', session_io.MODEL_FILTER)
        if not file_name:
            return False
        return self._write_model(pathlib.Path(file_name))

    def _write_model(self, path: pathlib.Path) -> bool:
        try:
            written = session_io.export_model(self._scene.model, path)
        except ValueError as error:
            # An incomplete graph is not a failure of the editor, it is something the user still has to do.
            problems = str(error).splitlines()
            self._report_error(
                'This model cannot be exported yet',
                'The following must be resolved first:\n\n  \u2022  ' + '\n  \u2022  '.join(problems),
            )
            return False
        except Exception as error:
            self._report_error('Unable to export the model', str(error))
            return False
        self._model_path = written
        self._update_document_state()
        self._statusBar.showMessage(f'Model exported to {written}')
        logging.getLogger('spark').log(MessageLevel.SUCCESS.value, f'Model exported to "{written}".')
        return True

    def _report_error(self, title: str, message: str) -> None:
        self._statusBar.showMessage(message.splitlines()[0])
        logging.getLogger('spark').error(f'{title}: {message}')
        QMessageBox.warning(self, title, message)

    def import_model_file(self) -> None:
        """
            Imports a model saved as a Spark configuration into the current graph.

            NOTE: Importing is additive and never replaces the graph. A file written by the editor carries the
            position of every module, which is honoured; models declared in code do not, and are laid out
            automatically.
        """
        model = self._scene.model
        if model.profile is None:
            return
        # Passing None as parent forces the OS native dialog.
        file_name, _ = QFileDialog.getOpenFileName(None, 'Import Model', '', 'Spark Config Files (*.scfg);;All Files (*)')
        if not file_name:
            return
        from spark.core.config import SparkConfig
        try:
            config = SparkConfig.from_file(file_name)
        except Exception as error:
            msg = f'Unable to read "{file_name}": {error}'
            self._statusBar.showMessage(msg)
            logging.getLogger('spark').error(msg)
            return
        # A model can only be expanded into the controller it was written for.
        source_profile = profile_for_config(config)
        if source_profile is not None and source_profile is not model.profile:
            msg = (
                f'"{pathlib.Path(file_name).name}" describes a {source_profile.label}, '
                f'it cannot be imported into a {model.profile.label}.'
            )
            self._statusBar.showMessage(msg)
            logging.getLogger('spark').error(msg)
            return
        self.view.import_config(config, label=pathlib.Path(file_name).stem)

    def open_preferences(self) -> None:
        dialog = PreferencesDialog(self)
        if dialog.exec():
            self.reload_ui()

    def reload_ui(self) -> None:
        # Save state
        model = self._scene.model
        selected_nodes = [item.model for item in self._scene.selectedItems() if isinstance(item, NodeItem)]
        # Recreate Scene & View
        new_scene = GraphScene(model)
        new_view = GraphView(new_scene)
        # NOTE: The canvas lives inside the stack that also holds the start screen, it is not the central
        # widget itself.
        self._stack.removeWidget(self.view)
        self._scene.deleteLater()
        self.view.deleteLater()
        self._stack.addWidget(new_view)
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
        self._update_document_state()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################