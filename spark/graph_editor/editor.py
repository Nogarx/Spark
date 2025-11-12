#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.widgets.dock_panel import QDockPanel

import os
import sys
import enum
import typing as tp
from PySide6 import QtWidgets, QtCore, QtGui
from spark.nn.brain import BrainConfig
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG
from spark.graph_editor.ui.menu_bar import MenuBar
from spark.graph_editor.ui.status_bar import StatusBar

# TODO: Initializer selection missing
# TODO: Allow to set the specific class of subconfigs.
# TODO: Allow to set optional configs to None in the inspector.

# NOTE: We use numpy to  manage dtypes. Jax sometimes tries to move data (?) to the GPU,
# which in turn slows down the editor unnecesarily.

# NOTE: All code that uses PySide6-QtAds must be wrapped in another script.
# Any direct import of the package in this file leads to a segmentation fault error
# due keyboard events propagation from CDockWidget to libxkbcommon since the 
# application is not properly initialized and so is XKB keymap.
# Wrapping the code avoids this due to python being lazy c:
from spark.graph_editor.window import EditorWindow
from spark.graph_editor.ui.graph_panel import GraphPanel
from spark.graph_editor.ui.nodes_panel import NodesPanel
from spark.graph_editor.ui.inspector_panel import InspectorPanel
from spark.graph_editor.ui.console_panel import ConsolePanel, MessageLevel

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class DockPanels(enum.Enum):
    GRAPH = enum.auto()
    INSPECTOR = enum.auto()
    NODES = enum.auto()
    CONSOLE = enum.auto()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkGraphEditor:

    def __init__(self) -> None:

        # Check if the editor was launched in the CLI or is using IPykernel
        self._is_interactive = 'ipykernel' in sys.modules
        if self._is_interactive:
            # Integrate Qt event loop to avoid the %gui qt in the terminal
            from IPython import get_ipython
            get_ipython().enable_gui('qt')

        # QApplication instance.
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        self._panels: dict[DockPanels, QDockPanel] = {}
        self._current_session_path = None
        self._current_model_path = None

    def launch(self) -> None:
        """
            Creates and shows the editor window without blocking.
            This method is safe to call multiple times.
        """
        #if __name__ == "__main__":
        # If a previous window exists, explicitly delete it (safe)
        if getattr(self, 'window', None):
            self.window.close()
            self.window.deleteLater()
            del self.window

        # Create base window.
        self.window = EditorWindow()
        self.window.windowClosed.connect(self.exit_editor)
        # Default layout
        self._setup_layout()
        # General style
        self.window.setStyleSheet(
            f"""
                color: {GRAPH_EDITOR_CONFIG.default_font_color};
            """
        )
        self.window.showMaximized()
        # Start loop
        self.app.exec_()

    def exit_editor(self,) -> None:
        """
            Exit editor.
        """
        self.app.quit()

    def closeEvent(self, event)-> None:
        """
            Overrides the default close event to check for unsaved changes.
        """
        if self._maybe_save():
            event.accept()
        else:
            event.ignore()

    def _setup_layout(self,) -> None:
        """
            Initialize the default window layout.
        """
        # Main panel
        graph_panel = GraphPanel(parent=self.window)
        self._panels[DockPanels.GRAPH] = graph_panel
        self.window.dock_manager.setCentralWidget(graph_panel)
        self.graph = graph_panel.graph
        #graph_panel._debug_model()

        # Console panel
        console_panel = ConsolePanel(parent=self.window)
        self._panels[DockPanels.CONSOLE] = console_panel
        self.window.add_dock_widget(GRAPH_EDITOR_CONFIG.console_panel_pos, console_panel)
        # Nodes panel
        nodes_panel = NodesPanel(self.graph, parent=self.window)
        self._panels[DockPanels.NODES] = nodes_panel
        self.window.add_dock_widget(GRAPH_EDITOR_CONFIG.nodes_panel_pos, nodes_panel)
        # Inspector panel
        inspector_panel = InspectorPanel(parent=self.window)
        self._panels[DockPanels.INSPECTOR] = inspector_panel
        self.window.add_dock_widget(GRAPH_EDITOR_CONFIG.inspector_panel_pos, inspector_panel)

        # Menu bar
        self.menu_bar = MenuBar(self)
        self.window.setMenuBar(self.menu_bar)
        # Status bar
        self.status_bar = StatusBar()
        self.window.setStatusBar(self.status_bar)

        # Setup events
        inspector_panel.broadcast_message.connect(console_panel.publish_message)
        graph_panel.broadcast_message.connect(console_panel.publish_message)
        graph_panel.graph.broadcast_message.connect(console_panel.publish_message)
        graph_panel.graph.node_selection_changed.connect(inspector_panel.on_selection_update)
        graph_panel.graph.nodes_deleted.connect(lambda: inspector_panel.set_node(None))
        graph_panel.graph.stateChanged.connect(lambda _: self._update_ui_state())

    def _maybe_save(self) -> bool:
        """
            Checks for unsaved changes and asks the user if they want to save.
            Returns True if the operation should proceed (user saved or discarded),
            and False if the operation should be cancelled.
        """
        if self.graph._is_modified:
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
            self._clear_session()

    def _clear_session(self,) -> None:
        self.graph.clear_session()
        self._current_session_path = None
        self._current_model_path = None
        self.graph._is_modified = False
        self._update_ui_state()
        self._panels[DockPanels.CONSOLE].clear()

    def save_session(self) -> bool:
        """
            Saves the current session to a Spark Graph Editor file.
        """
        if self._current_session_path is None:
            return self.save_session_as()
        else:
            try:
                self.graph.save_session(self._current_session_path)
                self.graph._is_modified = False
                self._update_ui_state()
                self._panels[DockPanels.CONSOLE].publish_message(
                    MessageLevel.SUCCESS, 
                    f'Session sucessfully saved to \"{self._current_session_path}\".'
                )
                return True
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not save file:\n{e}")
                self._panels[DockPanels.CONSOLE].publish_message(
                    MessageLevel.ERROR, 
                    f'Failed to save session to \"{self._current_session_path}\": {e}'
                )
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
            # Try to also set the model_path
            try:
                self._current_model_path = f'{self._current_session_path.strip('.sge')}.scfg'
            except:
                pass
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
                self._clear_session()
                self.graph.load_session(path)
                self._current_session_path = path
                self.graph._is_modified = False
                self._panels[DockPanels.CONSOLE].publish_message(
                    MessageLevel.SUCCESS, 
                    f'Session loaded sucessfully from \"{self._current_session_path}\".'
                )
                self._update_ui_state()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not load file:\n{e}")
                self._panels[DockPanels.CONSOLE].publish_message(
                    MessageLevel.ERROR, 
                    f'Failed to load session from \"{self._current_session_path}\": {e}'
                )

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
                self._clear_session()
                self._current_model_path = path
                config = BrainConfig.from_file(self._current_model_path)
                self.graph.load_from_model(config)
                self.graph._is_modified = False
                self._panels[DockPanels.CONSOLE].publish_message(
                    MessageLevel.SUCCESS, 
                    f'Model loaded sucessfully from \"{self._current_model_path}\".'
                )
                self._update_ui_state()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not load file:\n{e}")
                self._panels[DockPanels.CONSOLE].publish_message(
                    MessageLevel.ERROR, 
                    f'Failed to load model from \"{self._current_model_path}\": {e}'
                )

    def export_model(self) -> bool:
        """
            Exports the graph state to a Spark configuration file.
        """
        if self._current_model_path is None:
            return self.export_model_as()
        else:
            #try:
                # Validate model.
                errors = self.graph.validate_graph()
                if len(errors) > 0: 
                    QtWidgets.QMessageBox.warning(
                        self.graph.viewer(), 
                        'Invalid Model', 'The following errors were detected:\n\n  •  '+'\n\n  •  '.join(errors)+'\n'
                    )
                    for e in errors:
                        self._panels[DockPanels.CONSOLE].publish_message(
                            MessageLevel.ERROR, 
                            f'Failed to export model to \"{self._current_model_path}\": {e}'
                        )
                    return 
                
                brain_config = self.graph.build_brain_config()
                brain_config.to_file(self._current_model_path)
                self._panels[DockPanels.CONSOLE].publish_message(
                    MessageLevel.SUCCESS, 
                    f'Model exported sucessfully to \"{self._current_model_path}\".'
                )
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
                    f'The file \"{os.path.basename(path)}\" already exists.<br>Do you want to replace it?',
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No
                )
                if ret == QtWidgets.QMessageBox.StandardButton.No:
                    continue 
            self._current_model_path = path
            # Try to also set the session_path
            try:
                self._current_session_path = self._current_model_path.strip('.scfg') + '.sge'
            except:
                pass
            return self.export_model()
        
        return False

    def _update_ui_state(self) -> None:
        """
            Updates all state-dependent UI elements like titles and action labels.
        """
        # Update window title to show file name and modification status
        base_title = 'Spark Graph Editor'
        if self._current_session_path:
            file_name = os.path.basename(self._current_session_path)
            modified_marker = ' *' if self.graph._is_modified else ''
            self.window.setWindowTitle(f'{base_title} - {file_name if file_name else 'Untitled'}{modified_marker}')
        else:
            modified_marker = ' *' if self.graph._is_modified else ''
            self.window.setWindowTitle(f'{base_title} - Untitled{modified_marker}')
        self.menu_bar._on_graph_modified(self.graph._is_modified, self._current_model_path)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################