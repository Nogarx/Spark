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

.. py:class:: ConcatConfig

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterfaceConfig`


   Configuration for `Concat`.


.. py:class:: Concat(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Joins several inputs into one flat payload.

   Every input is flattened and concatenated in the order the ports were declared. All inputs
   must carry the same payload type, which is also the type of the result.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ConcatConfig, optional

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Any number of inputs, all of the same payload type. Named by the graph.

   :Output Ports: **output** (*SparkPayload*) -- The joined inputs, one dimensional and of the same payload type as the inputs.

   .. seealso::

      :py:obj:`ConcatReshape`
          The same join, followed by a reshape.


   .. py:attribute:: config
      :type:  ConcatConfig


   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: __call__(**inputs)

      Flattens and concatenates every input.

      :param \*\*inputs: Any number of inputs, all of the same payload type.
      :type \*\*inputs: SparkPayload

      :returns: Dictionary with one entry, ``output``, one dimensional and of the payload type of the
                inputs.
      :rtype: ControlInterfaceOutput



.. py:class:: ConcatReshapeConfig

   Bases: :py:obj:`ConcatConfig`


   Configuration for `ConcatReshape`.

   :param reshape: Shape of the result. Its size must match the total size of the inputs.
   :type reshape: tuple of int


   .. py:attribute:: reshape
      :type:  tuple[int, ...]


.. py:class:: ConcatReshape(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Joins several inputs into one payload of a given shape.

   `Concat` followed by a reshape to ``reshape``, which is what lets several one dimensional
   sources feed a module that expects a pool of a particular shape.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ConcatReshapeConfig

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Any number of inputs, all of the same payload type. Named by the graph.

   :Output Ports: **output** (*SparkPayload*) -- The joined inputs, of shape ``reshape`` and of the same payload type as the inputs.

   .. seealso::

      :py:obj:`Concat`
          The join, without the reshape.


   .. py:attribute:: config
      :type:  ConcatReshapeConfig


   .. py:attribute:: reshape
      :value: ()



   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:method:: __call__(**inputs)

      Flattens, concatenates and reshapes every input.

      :param \*\*inputs: Any number of inputs, all of the same payload type.
      :type \*\*inputs: SparkPayload

      :returns: Dictionary with one entry, ``output``, of shape ``reshape`` and of the payload type of
                the inputs.
      :rtype: ControlInterfaceOutput



