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
import pathlib
import typing as tp
from PySide6 import QtWidgets, QtCore, QtGui
from spark.nn.brain import BrainConfig
from spark.graph_editor.editor_config import GRAPH_EDITOR_CONFIG
from spark.graph_editor.ui.menu_bar import MenuBar
from spark.graph_editor.ui.status_bar import StatusBar

# TODO: Allow to set the specific class of subconfigs.
# TODO: Allow to set optional configs to None in the inspector.
# TODO: Allow basic shortcuts: ctrl+c, ctrl+v, etc.
# TODO: Create undo/redo stack (ctrl+z, ctrl+y).

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

import logging
logger = logging.getLogger('Spark')

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
        self._session_path = None
        self._model_path = None
        self._is_dirty = False



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
        # BUG: Part of the Ctrl+S workaround.
        self.window.editor = self
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
        #graph_panel.graph.on_update.connect(lambda _: self._update_ui_state())

        self.graph.on_update.connect(self._on_graph_update)
        self._panels[DockPanels.INSPECTOR].on_update.connect(self._on_inspector_update)

        # BUG: Patch because NodeGraph keeps overriding the Save shortcut >:|
        self.graph._viewer.BUGFIX_on_save.connect(self.save_session)

    def _maybe_save(self) -> bool:
        """
            Checks for unsaved changes and asks the user if they want to save.
            Returns True if the operation should proceed (user saved or discarded),
            and False if the operation should be cancelled.
        """

        if not self.graph._is_dirty:
            return True
        
        msg_box = QtWidgets.QMessageBox(None)
        msg_box.setText('The document has been modified.')
        msg_box.setInformativeText('Do you want to save your changes?')
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
        self._session_path = None
        self._model_path = None
        self._clear_dirty_flags()
        self._update_ui_state()
        self._panels[DockPanels.CONSOLE].clear()



    def save_session(self) -> bool:
        """
            Saves the current session to a Spark Graph Editor file.
        """
        if self._session_path is None:
            return self.save_session_as()
        else:
            try:
                brain_config = self.graph.build_brain_config(is_partial=True)
                brain_config.to_file(self._model_path, is_partial=True)
                self._clear_dirty_flags()
                self._update_ui_state()
                msg = f'Session sucessfully saved to \"{self._session_path}\".'
                self._panels[DockPanels.CONSOLE].publish_message(MessageLevel.SUCCESS, msg)
                logger.info(msg)
                return True
            except Exception as e:
                msg = f'Failed to save session to \"{self._session_path}\": {e}'
                QtWidgets.QMessageBox.critical(None, "Error", f"Could not save file:\n{e}")
                self._panels[DockPanels.CONSOLE].publish_message(MessageLevel.ERROR, msg)
                logger.error(msg)
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
            path = pathlib.Path(dialog.selectedFiles()[0])
            if path.exists():
                ret = QtWidgets.QMessageBox.question(
                    None,
                    'Confirm Overwrite',
                    f'The file "{path.name}" already exists.<br>Do you want to replace it?',
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No
                )
                if ret == QtWidgets.QMessageBox.StandardButton.No:
                    continue 
            self._session_path = path.with_suffix('.sge')
            # Try to also set the model_path
            if self._model_path is None:
                self._model_path = path.with_suffix('.scfg')
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
            path = pathlib.Path(path)
            try:
                self._clear_session()
                config = BrainConfig.from_file(path, is_partial=True)
                self.graph.load_from_model(config)
                self._clear_dirty_flags()
                self._panels[DockPanels.INSPECTOR].clear_selection()
                msg = f'Session loaded sucessfully from \"{path}\".'
                self._panels[DockPanels.CONSOLE].publish_message(MessageLevel.SUCCESS, msg)
                logger.info(msg)
                self._session_path = path.with_suffix('.sge')
                self._model_path = path.with_suffix('.scfg')
                self._update_ui_state()
            except Exception as e:
                msg = f'Failed to load session from \"{path}\": {e}'
                QtWidgets.QMessageBox.critical(None, 'Error', msg)
                self._panels[DockPanels.CONSOLE].publish_message(MessageLevel.ERROR, msg)
                logger.critical(msg)



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
            path = pathlib.Path(path)
            try:
                self._clear_session()
                config = BrainConfig.from_file(path, is_partial=False)
                self.graph.load_from_model(config)
                self._clear_dirty_flags()
                self._panels[DockPanels.INSPECTOR].clear_selection()
                msg = f'Model loaded sucessfully from \"{path}\".'
                self._panels[DockPanels.CONSOLE].publish_message(MessageLevel.SUCCESS, msg)
                logger.info(msg)
                self._session_path = path.with_suffix('.sge')
                self._model_path = path.with_suffix('.scfg')
                self._update_ui_state()
            except Exception as e:
                msg = f'Failed to load model from \"{path}\": {e}'
                QtWidgets.QMessageBox.critical(None, 'Error', msg)
                self._panels[DockPanels.CONSOLE].publish_message(MessageLevel.ERROR, msg)
                logger.critical(msg)



    def export_model(self) -> bool:
        """
            Exports the graph state to a Spark configuration file.
        """
        if self._model_path is None:
            return self.export_model_as()
        else:
            try:
                # Validate connectivity.
                errors = self.graph.validate_graph()
                if len(errors) > 0: 
                    QtWidgets.QMessageBox.warning(
                        self.graph.viewer(), 
                        'Invalid Model', 'The following errors were detected:\n\n  •  '+'\n\n  •  '.join(errors)+'\n'
                    )
                    for e in errors:
                        msg = f'Failed to export model to \"{self._model_path}\": {e}'
                        self._panels[DockPanels.CONSOLE].publish_message(MessageLevel.ERROR, msg)
                        logger.error(msg)
                    return False
                
                # Validate configuration.
                errors = []
                brain_config = self.graph.build_brain_config(is_partial=False, errors=errors)
                if len(errors) > 0: 
                    QtWidgets.QMessageBox.warning(
                        self.graph.viewer(), 
                        'Invalid Model', 'The following errors were detected:\n\n  •  '+'\n\n  •  '.join(f'{'/'.join(o)}: {e}' for o,e in errors)+'\n'
                    )
                    for e in errors:
                        # Validate errors are tuples (path, error)
                        msg = f'Failed to export model to \"{self._model_path}\": {f'{'/'.join(e[0])}: {e[1]}'}'
                        self._panels[DockPanels.CONSOLE].publish_message(MessageLevel.ERROR, msg)
                        logger.error(msg)
                    return False

                # Write file.
                try:
                    brain_config.to_file(self._model_path, is_partial=False)
                    msg = f'Model exported sucessfully to \"{self._model_path}\".'
                    self._panels[DockPanels.CONSOLE].publish_message(MessageLevel.SUCCESS, msg)
                    logger.info(msg)
                    return True
                except Exception as e:
                    msg = f'Failed to export model to \"{self._model_path}\": {e}'
                    self._panels[DockPanels.CONSOLE].publish_message(MessageLevel.ERROR, msg)
                    logger.error(msg)
                    return False
                
            except Exception as e:
                msg = f'Could not save file:\n{e}'
                QtWidgets.QMessageBox.critical(None, 'Error', msg)
                logger.critical(msg)
                return False



    def export_model_as(self) -> bool:
        """
            Exports the graph state to a new Spark configuration file.
        """
        dialog = QtWidgets.QFileDialog(None, 'Save Session As')
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter('Spark Cfg File (*.scfg);;All Files (*)')
        dialog.setDefaultSuffix('scfg')

        while dialog.exec():
            path = pathlib.Path(dialog.selectedFiles()[0])
            if path.exists():
                ret = QtWidgets.QMessageBox.question(
                    None,
                    'Confirm Overwrite',
                    f'The file \"{path.name}\" already exists.<br>Do you want to replace it?',
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No
                )
                if ret == QtWidgets.QMessageBox.StandardButton.No:
                    continue 
            self._model_path = path.with_suffix('.scfg')
            # Try to also set the session_path
            if self._session_path is None:
                self._session_path = path.with_suffix('.sge')
            return self.export_model()
        
        return False



    def _update_ui_state(self) -> None:
        """
            Updates all state-dependent UI elements like titles and action labels.
        """
        # Update window title to show file name and modification status
        base_title = 'Spark Graph Editor'
        file_name = self._session_path.stem  if self._session_path else 'Untitled'
        export_path = self._model_path.name if self._model_path else ''
        modified_marker = ' *' if self._is_dirty else ''
        self.window.setWindowTitle(f'{base_title} - {file_name}{modified_marker}')
        self.menu_bar._on_graph_modified(self._is_dirty, export_path)

    def _on_graph_update(self, *args, **kwargs) -> None:
        self._is_dirty = True
        self._update_ui_state()

    def _on_inspector_update(self, *args, **kwargs) -> None:
        self._is_dirty = True
        self._update_ui_state()

    def _clear_dirty_flags(self,) -> None:
        self.graph._is_dirty = False
        self._panels[DockPanels.INSPECTOR].set_dirty_flag(False)
        self._is_dirty = False
        self._update_ui_state()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################