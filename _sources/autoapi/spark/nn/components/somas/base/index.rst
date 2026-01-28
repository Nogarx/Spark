spark.nn.components.somas.base
==============================

.. py:module:: spark.nn.components.somas.base


Attributes
----------

.. autoapisummary::

   spark.nn.components.somas.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.components.somas.base.SomaOutput
   spark.nn.components.somas.base.SomaConfig
   spark.nn.components.somas.base.Soma


Module Contents
---------------

.. py:class:: SomaOutput

   Bases: :py:obj:`TypedDict`


   Generic soma model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: spikes
      :type:  spark.core.payloads.SpikeArray


   .. py:attribute:: potential
      :type:  spark.core.payloads.PotentialArray


.. py:class:: SomaConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Abstract soma model configuration class.


.. py:data:: ConfigT

.. py:class:: Soma(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract soma model.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets neuron states to their initial values.



   .. py:method:: __call__(current)

      Update neuron's states and compute spikes.



