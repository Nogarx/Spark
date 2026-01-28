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

.. py:class:: SamplerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterfaceConfig`


   Sampler configuration class.


   .. py:attribute:: sample_size
      :type:  int


.. py:class:: Sampler(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.control.base.ControlInterface`


   Sample a single input streams of inputs of the same type into a single stream.
   Indices are selected randomly and remain fixed.

   Init:
       sample_size: int

   Input:
       input: type[SparkPayload]

   Output:
       output: type[SparkPayload]


   .. py:attribute:: config
      :type:  SamplerConfig


   .. py:attribute:: sample_size


   .. py:method:: build(input_specs)

      Build method.



   .. py:property:: indices
      :type: jax.Array



   .. py:method:: __call__(inputs)

      Sub/Super-sample the input stream to get the pre-specified number of samples.



