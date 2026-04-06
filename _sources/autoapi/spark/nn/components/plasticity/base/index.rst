spark.nn.components.plasticity.base
===================================

.. py:module:: spark.nn.components.plasticity.base


Attributes
----------

.. autoapisummary::

   spark.nn.components.plasticity.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.components.plasticity.base.PlasticityOutput
   spark.nn.components.plasticity.base.PlasticityConfig
   spark.nn.components.plasticity.base.Plasticity


Module Contents
---------------

.. py:class:: PlasticityOutput

   Bases: :py:obj:`TypedDict`


   Generic plasticity rule model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: kernel
      :type:  spark.core.payloads.FloatArray


.. py:class:: PlasticityConfig(skip_validation = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Abstract plasticity rule configuration class.


.. py:data:: ConfigT

.. py:class:: Plasticity(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract plasticity rule model.


