spark.graph_editor.models.recent_files
======================================

.. py:module:: spark.graph_editor.models.recent_files


Attributes
----------

.. autoapisummary::

   spark.graph_editor.models.recent_files.logger
   spark.graph_editor.models.recent_files.SETTINGS_KEY
   spark.graph_editor.models.recent_files.MAX_RECENT


Functions
---------

.. autoapisummary::

   spark.graph_editor.models.recent_files.recent_files
   spark.graph_editor.models.recent_files.remember
   spark.graph_editor.models.recent_files.forget
   spark.graph_editor.models.recent_files.clear


Module Contents
---------------

.. py:data:: logger

.. py:data:: SETTINGS_KEY
   :value: 'recent_files'


.. py:data:: MAX_RECENT
   :value: 8


   Number of files kept.

.. py:function:: recent_files(existing_only = True)

   Files opened or written recently, most recent first.

   :param existing_only: Drop the entries that are no longer on disk.
   :type existing_only: bool, default True

   :returns: The remembered files.
   :rtype: list of pathlib.Path


.. py:function:: remember(path)

   Puts a file at the top of the list, moving it there if it was already known.


.. py:function:: forget(path)

   Drops a file from the list. The path is dropped both as given and as resolved.


.. py:function:: clear()

   Forgets every file.


