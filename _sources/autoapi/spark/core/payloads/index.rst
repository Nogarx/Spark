spark.core.payloads
===================

.. py:module:: spark.core.payloads


Classes
-------

.. autoapisummary::

   spark.core.payloads.SparkPayload
   spark.core.payloads.ValueSparkPayload
   spark.core.payloads.SpikeArray
   spark.core.payloads.CurrentArray
   spark.core.payloads.PotentialArray
   spark.core.payloads.BooleanMask
   spark.core.payloads.IntegerMask
   spark.core.payloads.FloatArray
   spark.core.payloads.IntegerArray


Module Contents
---------------

.. py:class:: SparkPayload

   Bases: :py:obj:`abc.ABC`


   Abstract payload definition to validate exchanges between SparkModule's.


   .. py:property:: shape
      :type: Any



   .. py:property:: dtype
      :type: Any



.. py:class:: ValueSparkPayload

   Bases: :py:obj:`SparkPayload`, :py:obj:`abc.ABC`


   Abstract payload definition to single value payloads.


   .. py:attribute:: value
      :type:  jax.numpy.ndarray


   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:property:: shape
      :type: spark.core.shape.Shape



   .. py:property:: dtype
      :type: jax.typing.DTypeLike



.. py:class:: SpikeArray

   Bases: :py:obj:`ValueSparkPayload`


   Representation of a collection of spike events.


.. py:class:: CurrentArray

   Bases: :py:obj:`ValueSparkPayload`


   Representation of a collection of currents.


.. py:class:: PotentialArray

   Bases: :py:obj:`ValueSparkPayload`


   Representation of a collection of membrane potentials.


.. py:class:: BooleanMask

   Bases: :py:obj:`ValueSparkPayload`


   Representation of an inhibitory boolean mask.


.. py:class:: IntegerMask

   Bases: :py:obj:`ValueSparkPayload`


   Representation of an integer mask.


.. py:class:: FloatArray

   Bases: :py:obj:`ValueSparkPayload`


   Representation of a float array.


.. py:class:: IntegerArray

   Bases: :py:obj:`ValueSparkPayload`


   Representation of an integer array.


