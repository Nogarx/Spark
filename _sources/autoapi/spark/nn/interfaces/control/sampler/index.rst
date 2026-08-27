spark.nn.interfaces.control.sampler
===================================

.. py:module:: spark.nn.interfaces.control.sampler


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.control.sampler.SamplerConfig
   spark.nn.interfaces.control.sampler.Sampler


Module Contents
---------------

.. py:class:: SamplerConfig

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterfaceConfig`


   Configuration for `Sampler`.

   :param sample_size: Shape of the result. May be larger than the input, in which case entries are drawn
                       more than once.
   :type sample_size: tuple of int


   .. py:attribute:: sample_size
      :type:  int


.. py:class:: Sampler(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Draws a fixed set of entries from its inputs.

   The inputs are flattened, concatenated and indexed by a set of indices drawn once at build
   time and held for the lifetime of the module. The same entries are read on every step, so
   this is a fixed projection rather than a fresh sample per step.

   Indices are drawn with replacement, so ``sample_size`` may exceed the size of the input
   and an entry may appear more than once.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: SamplerConfig

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Any number of inputs, all of the same payload type. Named by the graph.

   :Output Ports: **output** (*SparkPayload*) -- The drawn entries, of shape ``sample_size`` and of the same payload type as the inputs.

   :Properties: **indices** (*jax.Array*) -- Flat indices drawn at build time and read on every step. Read only.


   .. py:attribute:: config
      :type:  SamplerConfig


   .. py:attribute:: sample_size


   .. py:method:: build(**abc_args)

      Lazy initialize method.

      Called once, by `_build`, with the payloads of the first call. Used by any subclass of
      `SparkModule` to define any variable that depends on the shape of the input to the module.

      :param \*\*abc_args: The payloads that arrived on the first call, by port name.
      :type \*\*abc_args: SparkPayload



   .. py:property:: indices
      :type: jax.Array



   .. py:method:: __call__(**inputs)

      Reads the drawn entries out of the inputs.

      :param \*\*inputs: Any number of inputs, all of the same payload type.
      :type \*\*inputs: SparkPayload

      :returns: Dictionary with one entry, ``output``, of shape ``sample_size`` and of the payload
                type of the inputs.
      :rtype: ControlInterfaceOutput



