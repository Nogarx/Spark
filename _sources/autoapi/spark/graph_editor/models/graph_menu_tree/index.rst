spark.graph_editor.models.graph_menu_tree
=========================================

.. py:module:: spark.graph_editor.models.graph_menu_tree


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.graph_menu_tree.HierarchicalMenuTree


Module Contents
---------------

.. py:class:: HierarchicalMenuTree(graph_menu)

   A manager class for NodeGraphMenu's that allows for easy, hierarchical creation and access of submenus.

   Init:
       graph_menu: NodeGraphMenu, base NodeGraphMenu menu of the NodeGraphQt object


   .. py:attribute:: graph_menu_ref
      :type:  NodeGraphQt.NodeGraphMenu


   .. py:attribute:: children
      :type:  dict[str, HierarchicalMenuTree]


   .. py:method:: __getitem__(path)

      Get / Create a submenu with the given name.

      Input:
          path: str | list[str], path of the requested submenu

      Output:
          HierarchicalMenuTree, requested submenu



   .. py:method:: add_command(name, func, shortcut = None)

      Add a graph command to current selected menu on the hierarchy.

      Input:
          name: str, name for the new command
          func: tp.Callable, function associated with the new command
          shortcut: str, keyboard shortcut for the new command



