spark.nn.components.learning_rules.base
=======================================

.. py:module:: spark.nn.components.learning_rules.base


Attributes
----------

.. autoapisummary::

   spark.nn.components.learning_rules.base.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.components.learning_rules.base.LearningRuleOutput
   spark.nn.components.learning_rules.base.LearningRuleConfig
   spark.nn.components.learning_rules.base.LearningRule


Module Contents
---------------

.. py:class:: LearningRuleOutput

   Bases: :py:obj:`TypedDict`


   Generic learning rule model output spec.

   Initialize self.  See help(type(self)) for accurate signature.


   .. py:attribute:: kernel
      :type:  spark.core.payloads.FloatArray


.. py:class:: LearningRuleConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.ComponentConfig`


   Abstract learning rule configuration class.


.. py:data:: ConfigT

.. py:class:: LearningRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.base.Component`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Abstract learning rule model.


