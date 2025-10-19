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
   spark.nn.components.synapses.base.Synanpses


Module Contents
---------------

.. py:class:: SynanpsesOutput

   Bases: :py:obj:`TypedDict`


   Generic synapses model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: currents
      :type:  spark.core.payloads.CurrentArray


.. py:class:: SynanpsesConfig(**kwargs)

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Abstract synapse model configuration class.


.. py:data:: ConfigT

.. py:class:: Synanpses(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract synapse model.

   Init:

   Input:
       spikes: SpikeArray

   Output:
       currents: CurrentArray


   .. py:method:: get_kernel()
      :abstractmethod:



   .. py:method:: set_kernel(new_kernel)
      :abstractmethod:



   .. py:method:: __call__(spikes)

      Compute synanpse's currents.



