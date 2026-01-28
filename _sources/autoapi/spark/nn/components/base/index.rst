spark.nn.components.base
========================

.. py:module:: spark.nn.components.base


Attributes
----------

.. autoapisummary::

   spark.nn.components.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.components.base.ComponentConfig
   spark.nn.components.base.Component


Module Contents
---------------

.. py:class:: ComponentConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.core.config.SparkConfig`


   Abstract neuronal component configuration class.


.. py:data:: ConfigT

.. py:class:: Component(config = None, **kwargs)

   Bases: :py:obj:`spark.core.module.SparkModule`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract neuronal component.


   .. py:attribute:: config
      :type:  ConfigT


