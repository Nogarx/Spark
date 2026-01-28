#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import os
import enum
import typing as tp
from PySide6 import QtCore, QtWidgets, QtGui

if tp.TYPE_CHECKING:
    from spark.graph_editor.editor import SparkGraphEditor

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class MenuActions(enum.Enum):
    FILE_NEW = enum.auto()
    FILE_LOAD_SESSION = enum.auto()
    FILE_LOAD_MODEL = enum.auto()
    FILE_SAVE_SESSION = enum.auto()
    FILE_SAVE_SESSION_AS = enum.auto()
    FILE_EXPORT = enum.auto()
    FILE_EXPORT_AS = enum.auto()
    FILE_EXIT = enum.auto()
    WINDOWS_INSPECTOR = enum.auto()
    WINDOWS_NODE_LIST = enum.auto()
    WINDOWS_CONSOLE = enum.auto()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# NOTE: This is just an interface, we still let the main editor to handle all the logic.
class MenuBar(QtWidgets.QMenuBar):
    """
        Graph Editor menu bar.
    """

    def __init__(self, editor: SparkGraphEditor, **kwargs):
        super().__init__(**kwargs)

        # Menu bars.
        self.file_menu = self.addMenu('File')
        self.windows_menu = self.addMenu('Windows')
        self._actions: dict[MenuActions, QtGui.QAction] = {}

        # New actions.
        new_action =  QtGui.QAction(
            'New',
            shortcut=QtGui.QKeySequence.StandardKey.New,
            shortcutContext=QtCore.Qt.ShortcutContext.ApplicationShortcut,
        )
        self._actions[MenuActions.FILE_NEW] = new_action
        new_action.triggered.connect(editor.new_session)
        self.file_menu.addAction(new_action)
        self.file_menu.addSeparator()

        # Load actions.
        load_session_action = QtGui.QAction(
            'Load Session',
            shortcut=QtGui.QKeySequence.StandardKey.Open,
            shortcutContext=QtCore.Qt.ShortcutContext.ApplicationShortcut,
        )
        self._actions[MenuActions.FILE_LOAD_SESSION] = load_session_action
        load_session_action.triggered.connect(editor.load_session)
        self.file_menu.addAction(load_session_action)
        load_model_action = QtGui.QAction(
            'Load Model',
            shortcut=QtGui.QKeySequence('Ctrl+Shift+O'),
            shortcutContext=QtCore.Qt.ShortcutContext.ApplicationShortcut,
        )
        self._actions[MenuActions.FILE_LOAD_MODEL] = load_model_action
        load_model_action.triggered.connect(editor.load_from_model)
        self.file_menu.addAction(load_model_action)
        self.file_menu.addSeparator()

        # Save actions.
        save_session_action = QtGui.QAction(
            'Save Session', 
            shortcut=QtGui.QKeySequence.StandardKey.Save,
            shortcutContext=QtCore.Qt.ShortcutContext.ApplicationShortcut,
        )
        self._actions[MenuActions.FILE_SAVE_SESSION] = save_session_action
        save_session_action.triggered.connect(editor.save_session)
        self.file_menu.addAction(save_session_action)
        save_session_as_action = QtGui.QAction(
            'Save Session as ...',
            shortcut=QtGui.QKeySequence.StandardKey.SaveAs,
            shortcutContext=QtCore.Qt.ShortcutContext.ApplicationShortcut,
        )
        self._actions[MenuActions.FILE_SAVE_SESSION_AS] = save_session_as_action
        save_session_as_action.triggered.connect(editor.save_session_as)
        self.file_menu.addAction(save_session_as_action)
        self.file_menu.addSeparator()

        # Export actions.
        export_action = QtGui.QAction(
            'Export Model',
            shortcut=QtGui.QKeySequence('Ctrl+E'),
            shortcutContext=QtCore.Qt.ShortcutContext.ApplicationShortcut,
        )
        self._actions[MenuActions.FILE_EXPORT] = export_action
        export_action.triggered.connect(editor.export_model)
        self.file_menu.addAction(export_action)
        export_as_action = QtGui.QAction(
            'Export Model as...',
            shortcut=QtGui.QKeySequence('Ctrl+Shift+E'),
            shortcutContext=QtCore.Qt.ShortcutContext.ApplicationShortcut,
        )
        self._actions[MenuActions.FILE_EXPORT_AS] = export_as_action
        export_as_action.triggered.connect(editor.export_model_as)
        self.file_menu.addAction(export_as_action)
        self.file_menu.addSeparator()

        # Close actions.
        exit_action = QtGui.QAction(
            'Exit',
            shortcut=QtGui.QKeySequence.StandardKey.Quit,
            shortcutContext=QtCore.Qt.ShortcutContext.ApplicationShortcut,
        )
        self._actions[MenuActions.FILE_EXIT] = exit_action
        exit_action.triggered.connect(editor.exit_editor)
        self.file_menu.addAction(exit_action)

        # Docks
        from spark.graph_editor.editor import DockPanels

        inspector_action = QtGui.QAction('Inspector', checkable=True, checked=True)
        self._actions[MenuActions.WINDOWS_INSPECTOR] = inspector_action
        inspector_action.triggered.connect(lambda: editor.toggle_panel(DockPanels.INSPECTOR))
        editor._panels[DockPanels.INSPECTOR].visibilityChanged.connect(
            lambda value: inspector_action.setChecked(value)
        )
        self.windows_menu.addAction(inspector_action)
        
        nodes_action = QtGui.QAction('Node List', checkable=True, checked=True)
        self._actions[MenuActions.WINDOWS_NODE_LIST] = nodes_action
        nodes_action.triggered.connect(lambda: editor.toggle_panel(DockPanels.NODES))
        editor._panels[DockPanels.NODES].visibilityChanged.connect(
            lambda value: nodes_action.setChecked(value)
        )
        self.windows_menu.addAction(nodes_action)

        console_action = QtGui.QAction('Console', checkable=True, checked=True)
        self._actions[MenuActions.WINDOWS_CONSOLE] = console_action
        console_action.triggered.connect(lambda: editor.toggle_panel(DockPanels.CONSOLE))
        editor._panels[DockPanels.CONSOLE].visibilityChanged.connect(
            lambda value: console_action.setChecked(value)
        )
        self.windows_menu.addAction(console_action)

    def _on_graph_modified(self, is_modified: bool, export_path: str) -> None:
        # Save options.
        self._actions[MenuActions.FILE_SAVE_SESSION].setEnabled(not is_modified)

        # Update 'Export' action text
        if export_path:
            self._actions[MenuActions.FILE_EXPORT].setText(f'Export to {os.path.basename(export_path)}')
        else:
            self._actions[MenuActions.FILE_EXPORT].setText(f'Export Model')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################