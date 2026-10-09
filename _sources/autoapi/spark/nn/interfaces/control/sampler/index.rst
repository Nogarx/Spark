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

   :param sample_size: Number of entries in each output. May be larger than the input, in which case entries
                       are drawn more than once.
   :type sample_size: int
   :param num_outputs: Number of outputs, ``output_0`` to ``output_{num_outputs - 1}``.
   :type num_outputs: int, default 1
   :param disjoint: Draws the outputs from separate entries. While the input holds ``num_outputs *
                    sample_size`` entries or more, no entry is drawn twice; past that, every entry is drawn
                    as evenly as possible, and an output holds an entry twice only if ``sample_size``
                    exceeds the input. Otherwise every output is drawn on its own, with replacement.
   :type disjoint: bool, default False


   .. py:attribute:: sample_size
      :type:  int


   .. py:attribute:: num_outputs
      :type:  int


   .. py:attribute:: disjoint
      :type:  bool


.. py:class:: Sampler(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Draws fixed sets of entries from its inputs, one per output.

   The inputs are flattened, concatenated and indexed by sets of indices drawn once at build
   time and held for the lifetime of the module. The same entries are read on every step, so
   each output is a fixed projection rather than a fresh sample per step.

   By default the indices of every output are drawn on their own, with replacement, so
   ``sample_size`` may exceed the size of the input and an entry may appear more than once.
   With ``disjoint``, the outputs are drawn from separate entries, which splits the input into
   ``num_outputs`` populations.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: SamplerConfig

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Any number of inputs, all of the same payload type. Named by the graph.

   :Output Ports: **output_0, ..., output_{num_outputs - 1}** (*SparkPayload*) -- The drawn entries, of shape ``sample_size`` and of the same payload type as the inputs.

   :Properties: **indices** (*jax.Array*) -- Flat indices drawn at build time and read on every step, of shape
                ``(num_outputs, sample_size)``. Read only.


   .. py:attribute:: config
      :type:  SamplerConfig


   .. py:attribute:: sample_size


   .. py:attribute:: num_outputs


   .. py:attribute:: disjoint


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

      :returns: One entry per output, ``output_0`` to ``output_{num_outputs - 1}``, each of shape
                ``sample_size`` and of the payload type of the inputs.
      :rtype: dict of str to SparkPayload



