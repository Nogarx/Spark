spark.core.checkpoint
=====================

.. py:module:: spark.core.checkpoint


Attributes
----------

.. autoapisummary::

   spark.core.checkpoint.CHECKPOINT_EXTENSION


Classes
-------

.. autoapisummary::

   spark.core.checkpoint.Checkpointable


Module Contents
---------------

.. py:data:: CHECKPOINT_EXTENSION
   :value: '.spark'


   Extension of the files written by `Checkpointable.checkpoint`.

.. py:class:: Checkpointable

   Mixin for creating and loading model checkpoints, used by `SparkModule` and `Controller`.

   The file is a gzipped tar holding model configuration ``model.scfg``, input specifications
   (in the metadata), and the current model ``state``, written by orbax.

   Model using the mixin need to define a ``config`` and define a ``get_input_specs`` method,
   which provide the input specs the model was built with.


   .. py:method:: checkpoint(path, overwrite = False, verbose = True, sha256 = False)

      Saves the model to a ``.spark`` file.

      :param path: Where to write. ``.spark`` is added to the name when it does not end with it.
      :type path: str or path-like
      :param overwrite: Replace an existing file.
      :type overwrite: bool, default False
      :param verbose: Log where the file was written, and its SHA-256, which `from_checkpoint` can check.
      :type verbose: bool, default True
      :param sha256: Export the SHA-256 to a file, ``<file>.sha256`` as ``sha256sum``.
                     We recommend adding a sha256 file when sharing models with other people to prevent
                     final users from consuming tampered files.
      :type sha256: bool, default False

      :returns: The file written.
      :rtype: pathlib.Path

      :raises RuntimeError: When the file exists and ``overwrite`` is False, or the model cannot be saved.

      .. rubric:: Notes

      The file is written aside and moved in place once complete. With several processes, each
      process that calls it writes the file on its own.



   .. py:method:: from_checkpoint(path, safe = True, verbose = True, sha256 = None)
      :classmethod:


      Loads a model from a ``.spark`` file.

      :param path: File to read, with or without its ``.spark`` extension.
      :type path: str or path-like
      :param safe: Refuse a file whose configuration or state is not where expected, or that holds links.
      :type safe: bool, default True
      :param verbose: Log where the model was read from.
      :type verbose: bool, default True
      :param sha256: Matching SHA-256 of the file.
      :type sha256: str, optional

      :returns: A model of the class saved: ``cls`` or a subclass.
      :rtype: SparkModule or Controller

      :raises RuntimeError: When the file cannot be read, does not have the SHA-256 given, or holds a model other
          than a ``cls``.



