spark.nn.components.learning_rules.zenke_rule
=============================================

.. py:module:: spark.nn.components.learning_rules.zenke_rule


Classes
-------

.. autoapisummary::

   spark.nn.components.learning_rules.zenke_rule.ZenkeRuleConfig
   spark.nn.components.learning_rules.zenke_rule.ZenkeRule


Module Contents
---------------

.. py:class:: ZenkeRuleConfig(**kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRuleConfig`


   ZenkeRule configuration class.


   .. py:attribute:: async_spikes
      :type:  bool


   .. py:attribute:: pre_tau
      :type:  float | jax.Array


   .. py:attribute:: post_tau
      :type:  float | jax.Array


   .. py:attribute:: post_slow_tau
      :type:  float | jax.Array


   .. py:attribute:: target_tau
      :type:  float | jax.Array


   .. py:attribute:: a
      :type:  float | jax.Array


   .. py:attribute:: b
      :type:  float | jax.Array


   .. py:attribute:: c
      :type:  float | jax.Array


   .. py:attribute:: d
      :type:  float | jax.Array


   .. py:attribute:: p
      :type:  float | jax.Array


   .. py:attribute:: gamma
      :type:  float | jax.Array


.. py:class:: ZenkeRule(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.components.learning_rules.base.LearningRule`


   Zenke plasticy rule model. This model is an extension of the classic Hebbian Rule.

   Init:
       async_spikes: bool
       pre_tau: float | jax.Array
       post_tau: float | jax.Array
       post_slow_tau: float | jax.Array
       target_tau: float | jax.Array
       a: float | jax.Array
       b: float | jax.Array
       c: float | jax.Array
       d: float | jax.Array
       P: float | jax.Array
       gamma: float | jax.Array

   Input:
       pre_spikes: SpikeArray
       post_spikes: SpikeArray
       current_kernel: FloatArray

   Output:
       kernel: FloatArray


   .. py:attribute:: config
      :type:  ZenkeRuleConfig


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Resets component state.



   .. py:method:: __call__(pre_spikes, post_spikes, current_kernel)

      Computes and returns the next kernel update.



