spark.nn.components.delays.base
===============================

.. py:module:: spark.nn.components.delays.base


Attributes
----------

.. autoapisummary::

   spark.nn.components.delays.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.components.delays.base.DelaysOutput
   spark.nn.components.delays.base.DelaysConfig
   spark.nn.components.delays.base.Delays


Module Contents
---------------

.. py:class:: DelaysOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a delay model.

   .. attribute:: out_spikes

      Spikes emitted in the past, delivered on this step.

      :type: SpikeArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: out_spikes
      :type:  spark.core.payloads.SpikeArray


.. py:class:: DelaysConfig

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Base configuration for synaptic delay models.

   Carries no field of its own. Concrete models declare their own parameters.


.. py:data:: ConfigT

.. py:class:: Delays(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for synaptic delay models.

   A delay model buffers incoming spikes and releases each one after the number of steps its
   kernel entry holds. Subclasses provide `_push`, which stores the spikes of the current
   step, and `_gather`, which reads the spikes due on it.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: DelaysConfig

   :Input Ports: **in_spikes** (*SpikeArray*) -- Spikes emitted on this step.

   :Output Ports: **out_spikes** (*SpikeArray*) -- Spikes emitted in the past and due on this step.

   :Properties: **kernel** (*IntegerArray*) -- Delay of every entry, in steps rather than in ms. Writable.

   .. rubric:: Notes

   The delays are exposed as the writable ``kernel`` property, in steps rather than in ms.

   .. seealso::

      :py:obj:`NDelays`
          One delay per presynaptic unit.

      :py:obj:`N2NDelays`
          One delay per (postsynaptic, presynaptic) pair.


   .. py:method:: kernel()


   .. py:method:: reset()
      :abstractmethod:


      Resets component state.



   .. py:method:: __call__(in_spikes)
      :abstractmethod:


      Stores the incoming spikes and returns the ones due on this step.

      :param in_spikes: Spikes emitted on this step.
      :type in_spikes: SpikeArray

      :returns: Dictionary with one entry, ``out_spikes``, the spikes whose delay elapsed on this
                step.
      :rtype: DelaysOutput



