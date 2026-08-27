spark.graph_editor.models.graph_registry
========================================

.. py:module:: spark.graph_editor.models.graph_registry


Attributes
----------

.. autoapisummary::

   spark.graph_editor.models.graph_registry.logger
   spark.graph_editor.models.graph_registry.EDITOR_REGISTRY


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.graph_registry.GraphEditorRegistryNamespace
   spark.graph_editor.models.graph_registry.GraphEditorRegistry


Module Contents
---------------

.. py:data:: logger

.. py:class:: GraphEditorRegistryNamespace(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Create a collection of name/value pairs.

   Example enumeration:

   >>> class Color(Enum):
   ...     RED = 1
   ...     BLUE = 2
   ...     GREEN = 3

   Access them by:

   - attribute access:

     >>> Color.RED
     <Color.RED: 1>

   - value lookup:

     >>> Color(1)
     <Color.RED: 1>

   - name lookup:

     >>> Color['RED']
     <Color.RED: 1>

   Enumerations can be iterated over, and know how many members they have:

   >>> len(Color)
   3

   >>> list(Color)
   [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

   Methods can be added to enumerations, and members can have their own
   attributes -- see the documentation for details.


   .. py:attribute:: Components


   .. py:attribute:: Initializers


   .. py:attribute:: Interfaces


   .. py:attribute:: Neurons


.. py:class:: GraphEditorRegistry

   Bases: :py:obj:`spark.core.registry.Registry`


   Generic registry implementation.


   .. py:attribute:: Components
      :type:  spark.core.registry.SubRegistry


.. py:data:: EDITOR_REGISTRY

