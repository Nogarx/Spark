spark.graph_editor.models.controller_profile
============================================

.. py:module:: spark.graph_editor.models.controller_profile


Attributes
----------

.. autoapisummary::

   spark.graph_editor.models.controller_profile.logger
   spark.graph_editor.models.controller_profile.CONTROLLER_PROFILES
   spark.graph_editor.models.controller_profile.BRAIN_PROFILE
   spark.graph_editor.models.controller_profile.NEURON_PROFILE


Classes
-------

.. autoapisummary::

   spark.graph_editor.models.controller_profile.ControllerProfile


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.controller_profile.register_controller_profile
   spark.graph_editor.models.controller_profile.get_controller_profile
   spark.graph_editor.models.controller_profile.profile_for_config


Module Contents
---------------

.. py:data:: logger

.. py:class:: ControllerProfile

   Describes how the editor behaves while building a given controller.


   .. py:attribute:: key
      :type:  str


   .. py:attribute:: label
      :type:  str


   .. py:attribute:: icon
      :type:  str


   .. py:attribute:: summary
      :type:  str


   .. py:attribute:: controller_name
      :type:  str


   .. py:attribute:: palette_namespaces
      :type:  tuple[spark.core.registry.RegistryNamespace, ...]


   .. py:attribute:: atomic_namespaces
      :type:  tuple[spark.core.registry.RegistryNamespace, ...]


   .. py:attribute:: import_namespaces
      :type:  tuple[spark.core.registry.RegistryNamespace, ...]
      :value: ()



   .. py:attribute:: model_namespace
      :type:  spark.core.registry.RegistryNamespace | None
      :value: None



   .. py:attribute:: cache_based
      :type:  bool
      :value: False



   .. py:property:: registry_entry
      :type: spark.core.registry.RegistryEntry | None


      Registry entry of the controller backing this profile.


   .. py:property:: controller_cls
      :type: type[spark.nn.controllers.base.Controller] | None


      Controller class backing this profile.


   .. py:property:: config_cls
      :type: type[spark.core.config.SparkConfig] | None


      Configuration class produced by a graph built under this profile.


   .. py:method:: self_property_specs()

      Properties the controller exposes to its own modules (the "__self__" origin of a PortMap).



   .. py:method:: is_importable(namespace)

      True if members of the namespace are expanded into their own modules instead of being placed.



   .. py:method:: is_atomic(namespace)

      True if members of the namespace are placed as a single node instead of being expanded.



   .. py:method:: hosts(other)

      True if a model built under "other" belongs on this canvas as a single node.

      A hosted model stays whole. An imported one is expanded into the modules it is made of.



   .. py:method:: allows_cycle(module_cls = None)

      True if a self/backwards connection is legal under this profile.

      Cache based controllers read every input from the previous timestep, so any cycle is legal.
      Otherwise the target module must define a recurrent contract.



.. py:data:: CONTROLLER_PROFILES
   :type:  dict[str, ControllerProfile]

   Controller profiles available to the editor, keyed by profile key.

.. py:function:: register_controller_profile(profile)

   Registers a new controller profile.


.. py:function:: get_controller_profile(key)

   Returns a controller profile by key.


.. py:function:: profile_for_config(config)

   Resolves the profile of an existing controller configuration.


.. py:data:: BRAIN_PROFILE

.. py:data:: NEURON_PROFILE

