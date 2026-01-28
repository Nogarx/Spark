spark.nn.interfaces.control.concat
==================================

.. py:module:: spark.nn.interfaces.control.concat


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.control.concat.ConcatConfig
   spark.nn.interfaces.control.concat.Concat
   spark.nn.interfaces.control.concat.ConcatReshapeConfig
   spark.nn.interfaces.control.concat.ConcatReshape


Module Contents
---------------

.. py:class:: ConcatConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterfaceConfig`


   Concat configuration class.


   .. py:attribute:: num_inputs
      :type:  int


.. py:class:: Concat(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Combines several streams of inputs of the same type into a single stream.

   Init:
       num_inputs: int
       payload_type: type[SparkPayload]

   Input:
       input: type[SparkPayload]

   Output:
       output: type[SparkPayload]


   .. py:attribute:: config
      :type:  ConcatConfig


   .. py:attribute:: num_inputs


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: __call__(inputs)

      Merge all input streams into a single data output stream.



.. py:class:: ConcatReshapeConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`ConcatConfig`


   ConcatReshape configuration class.


   .. py:attribute:: reshape
      :type:  tuple[int, Ellipsis]


.. py:class:: ConcatReshape(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Combines several streams of inputs of the same type into a single stream.

   Init:
       num_inputs: int
       reshape: tuple[int, ...]
       payload_type: type[SparkPayload]

   Input:
       input: type[SparkPayload]

   Output:
       output: type[SparkPayload]


   .. py:attribute:: config
      :type:  ConcatReshapeConfig


   .. py:attribute:: reshape


   .. py:attribute:: num_inputs


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: __call__(inputs)

      Merge all input streams into a single data output stream. Output stream is reshape to match the pre-specified shape.



