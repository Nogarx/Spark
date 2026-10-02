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


   Base class for the values modules exchange.

   A payload names what a quantity is, not only how it is shaped. A port declares the payload
   it carries, and the framework refuses a connection between two ports of different types,
   so a current cannot be wired where a potential is expected.

   Every payload is registered as a pytree, so it crosses a jit boundary as data.

   .. seealso::

      :py:obj:`ValueSparkPayload`
          Payloads holding a single array.

      :py:obj:`SpikeArray`
          Spikes and their inhibition mask, bit-packed.


   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



   .. py:property:: shape
      :type: Any



   .. py:property:: dtype
      :type: Any



.. py:class:: SpikeArray(spikes, inhibition_mask = None, async_spikes = False)

   Bases: :py:obj:`SparkPayload`


   Spike events of a pool, with the sign of each unit.

   The spike bit and the inhibition bit of every unit are packed into one ``uint8`` array,
   so the two travel together and a downstream module cannot read one without the other.

   Inhibition schema
   Excitatory -> + or 0
   Inhibitory -> - or 1

   Encoding schema
   (Spike bit, Inhibition bit)
   0: (False, False) ->  0
   1: (True,  False) ->  1
   2: (False, True)  -> -0
   3: (True,  True)  -> -1

   :param spikes: Non-zero where a unit spiked.
   :type spikes: jax.Array
   :param inhibition_mask: True where a unit is inhibitory. Broadcast against ``spikes`` when it has fewer
                           dimensions. Defaults to all excitatory.
   :type inhibition_mask: BooleanMask or jax.Array or bool, optional
   :param async_spikes: Marks one entry per (target, origin) pair rather than one per origin.
   :type async_spikes: bool, default False

   .. attribute:: spikes

      The spike bit, as bool.

      :type: jax.Array

   .. attribute:: inhibition_mask

      The inhibition bit, as bool.

      :type: jax.Array

   .. attribute:: value

      The signed spikes: ``+1`` for an excitatory spike, ``-1`` for an inhibitory one, ``0``
      for no spike.

      :type: jax.Array

   .. rubric:: Notes

   ``async_spikes`` is set by the delay models that give every connection its own delay, such
   as `N2NDelays`. The shape then grows from ``(origin_units,)`` to
   ``(target_units, origin_units)``. A synapse model that means to accept both forms has to
   read the flag and sum over the origin axes only.


   .. py:attribute:: async_spikes
      :type:  bool
      :value: False



   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:method:: __eq__(other)


   .. py:property:: spikes
      :type: jax.Array



   .. py:property:: inhibition_mask
      :type: jax.Array



   .. py:property:: value
      :type: jax.Array



   .. py:property:: shape
      :type: tuple[int, ...]



   .. py:property:: dtype
      :type: jax.typing.DTypeLike



.. py:class:: ValueSparkPayload

   Bases: :py:obj:`SparkPayload`, :py:obj:`abc.ABC`


   Base class for payloads holding a single array.

   :param value: The array the payload carries.
   :type value: jax.Array


   .. py:attribute:: value
      :type:  jax.numpy.ndarray


   .. py:method:: __jax_array__()


   .. py:method:: __array__(dtype=None)


   .. py:method:: tree_flatten()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



   .. py:property:: shape
      :type: tuple[int, ...]



   .. py:property:: dtype
      :type: jax.typing.DTypeLike



.. py:class:: CurrentArray

   Bases: :py:obj:`ValueSparkPayload`


   Synaptic current, in pA.

   Produced by a synapse model and consumed by a soma.


.. py:class:: PotentialArray

   Bases: :py:obj:`ValueSparkPayload`


   Membrane potential, in mV.

   Exposed by a soma as its ``potential`` property. Whether it is measured from rest or in
   absolute mV depends on the soma model.


.. py:class:: BooleanMask

   Bases: :py:obj:`ValueSparkPayload`


   Boolean mask over the units of a pool.

   Used for the inhibition mask a `Neuron` hands to its modules.


.. py:class:: IntegerMask

   Bases: :py:obj:`ValueSparkPayload`


   Integer mask over the units of a pool.

   Used for the connection type of every synapse, as exposed by `Plasticity.synaptic_mask`.


.. py:class:: FloatArray

   Bases: :py:obj:`ValueSparkPayload`


   Array of floats, with no unit attached.

   Used for synaptic weights and for the signals that are neither currents nor potentials,
   such as the third factor of a modulated plasticity rule.


.. py:class:: IntegerArray

   Bases: :py:obj:`ValueSparkPayload`


   Array of integers, with no unit attached.

   Used for conduction delays, which are counted in steps.


