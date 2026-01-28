spark.nn.components
===================

.. py:module:: spark.nn.components


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/base/index
   /autoapi/spark/nn/components/delays/index
   /autoapi/spark/nn/components/learning_rules/index
   /autoapi/spark/nn/components/somas/index
   /autoapi/spark/nn/components/synapses/index


Classes
-------

.. autoapisummary::

   spark.nn.components.Component
   spark.nn.components.ComponentConfig


Package Contents
----------------

.. py:class:: Component(config = None, **kwargs)

   Bases: :py:obj:`spark.core.module.SparkModule`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract neuronal component.


   .. py:attribute:: config
      :type:  ConfigT


.. py:class:: ComponentConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.core.config.SparkConfig`


   Abstract neuronal component configuration class.


