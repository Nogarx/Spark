spark.nn.components.learning_rules.hebbian_rule
===============================================

.. py:module:: spark.nn.components.learning_rules.hebbian_rule


Classes
-------

.. autoapisummary::

   spark.nn.components.learning_rules.hebbian_rule.HebbianRuleConfig
   spark.nn.components.learning_rules.hebbian_rule.HebbianRule
   spark.nn.components.learning_rules.hebbian_rule.OjaRuleConfig
   spark.nn.components.learning_rules.hebbian_rule.OjaRule


Module Contents
---------------

.. py:class:: HebbianRuleConfig(**kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRuleConfig`


   HebbianRule configuration class.


   .. py:attribute:: async_spikes
      :type:  bool


   .. py:attribute:: pre_tau
      :type:  float | jax.Array


   .. py:attribute:: post_tau
      :type:  float | jax.Array


   .. py:attribute:: gamma
      :type:  float | jax.Array


.. py:class:: HebbianRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRule`


   Hebbian plasticy rule model.

   Init:
       async_spikes: bool
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       gamma: float | jax.Array

   Input:
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       current_kernel: FloatArray

   Output:
       kernel: FloatArray


   .. py:attribute:: config
      :type:  HebbianRuleConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(pre_spikes, post_spikes, current_kernel)

      Computes and returns the next kernel update.



.. py:class:: OjaRuleConfig(**kwargs)

   Bases: :py:obj:`HebbianRuleConfig`


   HebbianRule configuration class.


   .. py:attribute:: stabilization_factor
      :type:  bool


.. py:class:: OjaRule(config = None, **kwargs)

   Bases: :py:obj:`HebbianRule`


   Oja's plasticy rule model.

   Init:
       async_spikes: bool
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       gamma: float | jax.Array

   Input:
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       current_kernel: FloatArray

   Output:
       kernel: FloatArray


