#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import enum
import typing as tp
import dataclasses as dc
from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
from PySide6.QtCore import Signal, QPoint

import logging
import spark.core.utils as utils
from spark.core.registry import REGISTRY, RegistryEntry, RegistryNamespace
from spark.graph_editor.models.node_model import SourceNodeModel, SinkNodeModel
from spark.graph_editor.models.node_factory import NODE_REGISTRY
from spark.graph_editor.models.controller_profile import ControllerProfile

logger = logging.getLogger('spark')

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
    Import = enum.auto()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass
class ActionData:
    command: ContextMenuCommand
    cls: type | None = None
    entry: RegistryEntry | None = None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class GraphContextMenu(QMenu):
    """
        Hierarchical context menu for adding nodes to the graph.
    """
    node_selected = Signal(RegistryEntry)

    def __init__(self, profile: ControllerProfile | None = None, parent=None) -> None:
        super().__init__(parent=parent)
        self._profile = profile
        self._build_menu()

    def set_profile(self, profile: ControllerProfile | None) -> None:
        """
            Rebuilds the palette for a controller profile.
        """
        if profile is self._profile:
            return
        self._profile = profile
        self.clear()
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

        create_source_action = QAction('Source Node', self)
        create_source_action.setData(ActionData(command=ContextMenuCommand.Create, cls=SourceNodeModel))
        interfaces_submenu.addAction(create_source_action)
        create_sink_action = QAction('Sink Node', self)
        create_sink_action.setData(ActionData(command=ContextMenuCommand.Create, cls=SinkNodeModel))
        interfaces_submenu.addAction(create_sink_action)
        interfaces_submenu.addSeparator()

        # Populate the palette declared by the active controller profile.
        # Which modules can be placed depends on the controller being built: a Brain hosts neurons and
        # interfaces, a Neuron hosts the components a neuron is made of.
        SUBMENU_MAX_DEPTH = 2
        namespaces = self._profile.palette_namespaces if self._profile else ()
        for namespace in namespaces:
            for key, entry in getattr(REGISTRY, namespace.name).items():
                # Controllers are the graph itself, they are never placed as a module.
                if len(entry.path) > 0 and entry.path[0].lower() == 'controller':
                    continue
                node_cls = NODE_REGISTRY.get(entry.get_cls())
                if node_cls is None:
                    logger.warning(f'No node model available for "{entry.name}", it will not be offered.')
                    continue
                path = entry.path[:SUBMENU_MAX_DEPTH]
                submenu = self._get_submenu(path)
                action = QAction(utils.to_human_readable(entry.get_cls().__name__), submenu)
                action.setData(ActionData(command=ContextMenuCommand.Create, cls=node_cls))
                submenu.addAction(action)

        # Models that are expanded instead of placed.
        # A registered model is a controller of its own. Under a profile that hosts it (a Neuron inside a
        # Brain) it belongs to the palette above and is placed as a single node. Under a profile that is that
        # controller it cannot be a node, and importing it adds the modules it is made of.
        import_namespaces = self._profile.import_namespaces if self._profile else ()
        if import_namespaces:
            import_submenu = QMenu('Import Model', self)
            self.addMenu(import_submenu)
            for namespace in import_namespaces:
                for key, entry in getattr(REGISTRY, namespace.name).items():
                    action = QAction(utils.to_human_readable(entry.get_cls().__name__), import_submenu)
                    action.setData(ActionData(command=ContextMenuCommand.Import, entry=entry))
                    import_submenu.addAction(action)
            self.addSeparator()

        # Common actions
        self.undo_action = QAction('Undo', self)
        self.undo_action.setData(ActionData(command=ContextMenuCommand.Undo))
        self.addAction(self.undo_action)
        self.redo_action = QAction('Redo', self)
        self.redo_action.setData(ActionData(command=ContextMenuCommand.Redo))
        self.addAction(self.redo_action)
        self.addSeparator()
        self.copy_action = QAction('Copy', self)
        self.copy_action.setData(ActionData(command=ContextMenuCommand.Copy))
        self.addAction(self.copy_action)
        self.paste_action = QAction('Paste', self)
        self.paste_action.setData(ActionData(command=ContextMenuCommand.Paste))
        self.addAction(self.paste_action)
        self.delete_action = QAction('Delete', self)
        self.delete_action.setData(ActionData(command=ContextMenuCommand.Delete))
        self.addAction(self.delete_action)

    def _get_submenu(self, path) -> QMenu:
        """
            Returns the submenu at a path, creating the levels it is missing.
        """
        submenu = self
        target_menu = self
        for name in path:
            submenu_name = utils.to_human_readable(name)
            target_menu = None
            for action in submenu.actions():
                menu: QMenu | None = action.menu()
                if menu is not None and menu.title() == submenu_name:
                    target_menu = menu
                    break
            if target_menu is None:
                target_menu = QMenu(submenu_name, submenu) 
                submenu.addMenu(target_menu)
            submenu = target_menu
        return target_menu


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
