#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import enum
import typing as tp
import dataclasses as dc
from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
from PySide6.QtCore import Signal, QPoint

import spark.core.utils as utils
from spark.core.registry import REGISTRY, RegistryEntry
from spark.graph_editor.models.node_model import SourceNodeModel, SinkNodeModel
from spark.graph_editor.models.node_factory import NODE_REGISTRY

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ContextMenuCommand(enum.Enum):
    Undo = enum.auto()
    Redo = enum.auto()
    Create = enum.auto()
    Delete = enum.auto()
    Copy = enum.auto()
    Paste = enum.auto()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass
class ActionData:
    command: ContextMenuCommand
    cls: type | None = None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class GraphContextMenu(QMenu):
    """
        Hierarchical context menu for adding nodes to the graph.
    """
    node_selected = Signal(RegistryEntry)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self._build_menu()

    def update_menu_state(self, has_selection: bool, can_undo: bool, can_redo: bool, can_paste: bool):
        self.copy_action.setEnabled(has_selection)
        self.paste_action.setEnabled(can_paste)
        self.delete_action.setEnabled(has_selection)
        self.undo_action.setEnabled(can_undo)
        self.redo_action.setEnabled(can_redo)

    def _build_menu(self) -> None:
        """
            Populate the menu from the core and editor registries.
        """

        # Submenus
        interfaces_submenu = QMenu('Interfaces', self) 
        self.addMenu(interfaces_submenu)

        # Source Node
        create_source_action = QAction('Source Node', self)
        create_source_action.setData(ActionData(command=ContextMenuCommand.Create, cls=SourceNodeModel))
        interfaces_submenu.addAction(create_source_action)
        # Sink Node
        create_sink_action = QAction('Sink Node', self)
        create_sink_action.setData(ActionData(command=ContextMenuCommand.Create, cls=SinkNodeModel))
        interfaces_submenu.addAction(create_sink_action)
        # Separator
        interfaces_submenu.addSeparator()

        # Populate components submenu
        SUBMENU_MAX_DEPTH = 2
        for key, entry in REGISTRY.Components.items():
            # NOTE: Skip controllers. Need to be done more gracefully.
            if entry.path[0].lower() == 'controller':
                continue
            path = entry.path[:SUBMENU_MAX_DEPTH]
            # Get submenu
            submenu = self._get_submenu(path)
            # Add action to submenu
            action = QAction(utils.to_human_readable(entry.get_cls().__name__), submenu)
            action.setData(ActionData(command=ContextMenuCommand.Create, cls=NODE_REGISTRY.get(entry.get_cls())))
            submenu.addAction(action)

        # Populate interfaces submenu
        for key, entry in REGISTRY.Interfaces.items():
            path = entry.path[:SUBMENU_MAX_DEPTH]
            # Get submenu
            submenu = self._get_submenu(path)
            # Add action to submenu
            action = QAction(utils.to_human_readable(entry.get_cls().__name__), submenu)
            action.setData(ActionData(command=ContextMenuCommand.Create, cls=NODE_REGISTRY.get(entry.get_cls())))
            submenu.addAction(action)

        # Common actions
        # Undo
        self.undo_action = QAction('Undo', self)
        self.undo_action.setData(ActionData(command=ContextMenuCommand.Undo))
        self.addAction(self.undo_action)
        # Redo
        self.redo_action = QAction('Redo', self)
        self.redo_action.setData(ActionData(command=ContextMenuCommand.Redo))
        self.addAction(self.redo_action)
        # Separator
        self.addSeparator()
        # Copy
        self.copy_action = QAction('Copy', self)
        self.copy_action.setData(ActionData(command=ContextMenuCommand.Copy))
        self.addAction(self.copy_action)
        # Paste
        self.paste_action = QAction('Paste', self)
        self.paste_action.setData(ActionData(command=ContextMenuCommand.Paste))
        self.addAction(self.paste_action)
        # Delete
        self.delete_action = QAction('Delete', self)
        self.delete_action.setData(ActionData(command=ContextMenuCommand.Delete))
        self.addAction(self.delete_action)

    def _get_submenu(self, path) -> QMenu:
        """
            Iterates searchs and construct submenus.
        """
        # Start on root
        submenu = self
        for name in path:
            submenu_name = utils.to_human_readable(name)
            # Check if submenu exists
            target_menu = None
            for action in submenu.actions():
                menu: QMenu | None = action.menu()
                if menu is not None and menu.title() == submenu_name:
                    target_menu = menu
                    break
            # Create submenu if it does not exits
            if target_menu is None:
                target_menu = QMenu(submenu_name, submenu) 
                submenu.addMenu(target_menu)
            submenu = target_menu
        # Return reference
        return target_menu


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
