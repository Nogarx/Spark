spark.nn.interfaces.control.base
================================

.. py:module:: spark.nn.interfaces.control.base


Attributes
----------

.. autoapisummary::

   spark.nn.interfaces.control.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.control.base.ControlInterfaceOutput
   spark.nn.interfaces.control.base.ControlInterfaceConfig
   spark.nn.interfaces.control.base.ControlInterface


Module Contents
---------------

.. py:class:: ControlInterfaceOutput

   Bases: :py:obj:`TypedDict`


   ControlInterface model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: output
      :type:  spark.core.payloads.SparkPayload


.. py:class:: ControlInterfaceConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.InterfaceConfig`


   Abstract ControlInterface model configuration class.


.. py:data:: ConfigT

.. py:class:: ControlInterface(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.Interface`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract ControlInterface model.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Control operation.



