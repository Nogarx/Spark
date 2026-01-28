spark.nn.interfaces.input.linear
================================

.. py:module:: spark.nn.interfaces.input.linear


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.input.linear.LinearSpikerConfig
   spark.nn.interfaces.input.linear.LinearSpiker


Module Contents
---------------

.. py:class:: LinearSpikerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterfaceConfig`


   LinearSpiker model configuration class.


   .. py:attribute:: tau
      :type:  float


   .. py:attribute:: cd
      :type:  float


   .. py:attribute:: max_freq
      :type:  float


.. py:class:: LinearSpiker(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterface`


   Transforms a continuous signal to a spiking signal.
   This transformation assumes a very simple linear neuron model without any type of adaptation or plasticity.
   Units have a fixed refractory period and at maximum input signal will fire up to some fixed frequency.

   Init:
       tau: float [ms]
       cd: float [ms]
       max_freq: float [Hz]

   Input:
       signal: FloatArray

   Output:
       spikes: SpikeArray


   .. py:attribute:: config
      :type:  LinearSpikerConfig


   .. py:attribute:: tau


   .. py:attribute:: cd


   .. py:attribute:: max_freq


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Reset module to its default state.



   .. py:method:: __call__(signal)

      Input interface operation.

      Input:
          A FloatArray of values in the range [0,1].
      Output:
          A SpikeArray of the same shape as the input.



