spark.core.payloads
===================

.. py:module:: spark.core.payloads


Classes
-------

.. autoapisummary::

   spark.core.payloads.SparkPayload
   spark.core.payloads.SpikeArray
   spark.core.payloads.ValueSparkPayload
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


   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



   .. py:property:: shape
      :type: Any



   .. py:property:: dtype
      :type: Any



.. py:class:: SpikeArray(spikes, inhibition_mask = False, async_spikes = False)

   Bases: :py:obj:`SparkPayload`


   Representation of a collection of spike events.

   Init:
       spikes: jax.Array[bool], True if neuron spiked, False otherwise
       inhibition_mask: jax.Array[bool], True if neuron is inhibitory, False otherwise

   The async_spikes flag is automatically set True by delay mechanisms that perform neuron-to-neuron specific delays.
   Note that when async_spikes is True the shape of the spikes changes from (origin_units,) to (origin_units, target_units).
   This is important when implementing new synaptic models, since fully valid synaptic models should be able to handle both cases.


   .. py:attribute:: async_spikes
      :type:  bool
      :value: False



   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:property:: spikes
      :type: jax.Array



   .. py:property:: inhibition_mask
      :type: jax.Array



   .. py:property:: value
      :type: jax.Array



   .. py:property:: shape
      :type: tuple[int, Ellipsis]



   .. py:property:: dtype
      :type: jax.typing.DTypeLike



.. py:class:: ValueSparkPayload

   Bases: :py:obj:`SparkPayload`, :py:obj:`abc.ABC`


   Abstract payload definition to single value payloads.


   .. py:attribute:: value
      :type:  jax.numpy.ndarray


   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



   .. py:property:: shape
      :type: tuple[int, Ellipsis]



   .. py:property:: dtype
      :type: jax.typing.DTypeLike



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


