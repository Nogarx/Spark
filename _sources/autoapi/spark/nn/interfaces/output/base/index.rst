spark.nn.interfaces.output.base
===============================

.. py:module:: spark.nn.interfaces.output.base


Attributes
----------

.. autoapisummary::

   spark.nn.interfaces.output.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.output.base.OutputInterfaceOutput
   spark.nn.interfaces.output.base.OutputInterfaceConfig
   spark.nn.interfaces.output.base.OutputInterface


Module Contents
---------------

.. py:class:: OutputInterfaceOutput

   Bases: :py:obj:`TypedDict`


   OutputInterface model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: signal
      :type:  spark.core.payloads.FloatArray


.. py:class:: OutputInterfaceConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.InterfaceConfig`


   Abstract OutputInterface model configuration class.


.. py:data:: ConfigT

.. py:class:: OutputInterface(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.base.Interface`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract OutputInterface model.


   .. py:method:: __call__(*args, **kwargs)
      :abstractmethod:


      Transform incomming spikes into a output signal.



