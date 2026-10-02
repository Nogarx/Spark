spark.nn.components.synapses.base
=================================

.. py:module:: spark.nn.components.synapses.base


Attributes
----------

.. autoapisummary::

   spark.nn.components.synapses.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.components.synapses.base.SynanpsesOutput
   spark.nn.components.synapses.base.SynanpsesConfig
   spark.nn.components.synapses.base.Synapses


Module Contents
---------------

.. py:class:: SynanpsesOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a synapse model.

   .. attribute:: currents

      Current delivered to each postsynaptic unit.

      :type: CurrentArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: currents
      :type:  spark.core.payloads.CurrentArray


.. py:class:: SynanpsesConfig

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Base configuration for synapse models.

   Carries no field of its own. Concrete models declare their own parameters.


.. py:data:: ConfigT

.. py:class:: Synapses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for synapse models.

   A synapse model turns presynaptic spikes into postsynaptic current. Subclasses provide
   `_dot`, which is the whole of the step.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: SynanpsesConfig, optional

   :Input Ports: **spikes** (*SpikeArray*) -- Presynaptic spikes.

   :Output Ports: **currents** (*CurrentArray*) -- Current delivered to each postsynaptic unit.

   :Properties: **kernel** (*FloatArray*) -- Synaptic weights, in pA. Writable, which is how a plasticity rule updates them.

   .. rubric:: Notes

   Kernel entries are in pA. The framework runs in half precision by default, and nA-scale
   weights lose too much of the mantissa to be summed reliably.

   The weights are exposed as the writable ``kernel`` property, which is what lets a
   plasticity rule read them and write them back.

   .. seealso::

      :py:obj:`LinearSynapses`
          Weighted sum of the incoming spikes.

      :py:obj:`TracedSynapses`
          Weighted sum filtered by a single exponential.


   .. py:method:: kernel()


   .. py:method:: get_kernel()
      :abstractmethod:



   .. py:method:: set_kernel(new_kernel)
      :abstractmethod:



   .. py:method:: __call__(spikes)

      Converts presynaptic spikes into postsynaptic current.

      :param spikes: Presynaptic spikes.
      :type spikes: SpikeArray

      :returns: Dictionary with one entry, ``currents``, the current delivered to each postsynaptic
                unit.
      :rtype: SynanpsesOutput



