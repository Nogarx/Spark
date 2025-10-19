#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import typing as tp
import spark.core.utils as utils
from NodeGraphQt import NodeGraphMenu

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class HierarchicalMenuTree:
    """
        A manager class for NodeGraphMenu's that allows for easy, hierarchical creation and access of submenus.

        Init:
            graph_menu: NodeGraphMenu, base NodeGraphMenu menu of the NodeGraphQt object
    """

    def __init__(self, graph_menu: NodeGraphMenu) -> None:
        # Sanity checks.
        if not isinstance(graph_menu, NodeGraphMenu):
            raise ValueError('HierarchicalMenuTree must be initialized with a valid NodeGraphMenu object.')
        # NodeGraphQt menu object.
        self.graph_menu_ref: NodeGraphMenu = graph_menu
        # Child MenuTree nodes.
        self.children: dict[str, 'HierarchicalMenuTree'] = {}

    def __getitem__(self, path: str | list[str]) -> 'HierarchicalMenuTree':
        """
            Get / Create a submenu with the given name.

            Input:
                path: str | list[str], path of the requested submenu
                
            Output:
                HierarchicalMenuTree, requested submenu 
        """
        # Sanity checks.
        if not isinstance(path, (list, str)):
            raise TypeError(f'\"path\" must be of type \"str | list[str]\" but got type \"{type(path).__name__}\".')
        if isinstance(path, list):
            for subpath in path:
                if not isinstance(subpath, str):
                    raise TypeError(f'Every element of \"path\" must be of type \"str\" but got type \"{type(subpath).__name__}\".')
        name = path.pop(0) if isinstance(path, list) else path
        # Check if we have already created and stored this child submenu.
        if name not in self.children:
            # Create the NodeGraphMenu submenu
            graph_submenu = self.graph_menu_ref.add_menu(name)
            self.children[name] = HierarchicalMenuTree(graph_submenu)
        # Return the requested child node.
        if isinstance(path, list) and len(path) > 0:
            return self.children[name][path]
        else:
            return self.children[name]

    def add_command(self, name: str, func: tp.Callable, shortcut: str = None) -> None:
        """
            Add a graph command to current selected menu on the hierarchy.

            Input:
                name: str, name for the new command
                func: tp.Callable, function associated with the new command
                shortcut: str, keyboard shortcut for the new command
        """
        # Sanity checks.
        if not callable(func):
            raise TypeError(
                f'Expected \"func\" to be callable, but \"{func}\" does not define a \"__call__\" method.'
            )
        if not isinstance(name, str):
            raise TypeError(
                f'Expected \"name\" to be of type \"str\", but got \"{name}\".'
            )
        if shortcut and not isinstance(shortcut, str):
            raise TypeError(
                f'Expected \"shortcut\" to be of type \"str | None\", but got \"{name}\".'
            )
        self.graph_menu_ref.add_command(utils.to_human_readable(name), func, shortcut)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
