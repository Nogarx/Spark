spark.nn.components.learning_rules.three_factor_rule
====================================================

.. py:module:: spark.nn.components.learning_rules.three_factor_rule


Classes
-------

.. autoapisummary::

   spark.nn.components.learning_rules.three_factor_rule.ThreeFactorHebbianRuleConfig
   spark.nn.components.learning_rules.three_factor_rule.ThreeFactorHebbianRule


Module Contents
---------------

.. py:class:: ThreeFactorHebbianRuleConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRuleConfig`


   ThreeFactorHebbianRule configuration class.


   .. py:attribute:: pre_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: post_tau
      :type:  float | jax.Array | spark.nn.initializers.base.Initializer


   .. py:attribute:: gamma
      :type:  float | jax.Array


.. py:class:: ThreeFactorHebbianRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRule`


   Three-factor Hebbian plasticy rule model.

   Init:
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       gamma: float | jax.Array

   Input:
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       kernel: FloatArray

   Output:
       kernel: FloatArray


   .. py:attribute:: config
      :type:  ThreeFactorHebbianRuleConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(reward, pre_spikes, post_spikes, kernel)

      Computes and returns the next kernel update.



