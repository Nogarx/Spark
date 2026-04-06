spark.graph_editor.models.utils
===============================

.. py:module:: spark.graph_editor.models.utils


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.utils.flattify_controller_config
   spark.graph_editor.models.utils.unflattify_controller_config


Module Contents
---------------

.. py:function:: flattify_controller_config(config)

   Generate a Config subclass programmatically, recursively building nested Configs from a Controller Config.


.. py:function:: unflattify_controller_config(cls, flat_config)

   Generate a ControllerConfig subclass programmatically, recursively building the modules_specs list
   from a another Config spec and a ControllerConfig template.


