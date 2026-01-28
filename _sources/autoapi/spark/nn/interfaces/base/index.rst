spark.nn.interfaces.base
========================

.. py:module:: spark.nn.interfaces.base


Attributes
----------

.. autoapisummary::

   spark.nn.interfaces.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.base.InterfaceOutput
   spark.nn.interfaces.base.InterfaceConfig
   spark.nn.interfaces.base.Interface


Module Contents
---------------

.. py:class:: InterfaceOutput

   Bases: :py:obj:`TypedDict`


   Generic Interface model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:class:: InterfaceConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.core.config.SparkConfig`


   Abstract Interface model configuration class.


.. py:data:: ConfigT

.. py:class:: Interface(config = None, **kwargs)

   Bases: :py:obj:`spark.core.module.SparkModule`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract Interface model.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Computes the control flow operation.



