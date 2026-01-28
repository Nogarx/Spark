spark.nn.interfaces.input.poisson
=================================

.. py:module:: spark.nn.interfaces.input.poisson


Classes
-------

.. autoapisummary::

   spark.nn.interfaces.input.poisson.PoissonSpikerConfig
   spark.nn.interfaces.input.poisson.PoissonSpiker


Module Contents
---------------

.. py:class:: PoissonSpikerConfig(__skip_validation__ = False, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterfaceConfig`


   PoissonSpiker model configuration class.


   .. py:attribute:: max_freq
      :type:  float


.. py:class:: PoissonSpiker(config = None, **kwargs)

   Bases: :py:obj:`spark.nn.interfaces.input.base.InputInterface`


   Transforms a continuous signal to a spiking signal.
   This transformation assumes a very simple poisson neuron model without any type of adaptation or plasticity.

   Init:
       max_freq: float [Hz]

   Input:
       signal: FloatArray

   Output:
       spikes: SpikeArray


   .. py:attribute:: config
      :type:  PoissonSpikerConfig


   .. py:attribute:: max_freq


   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: __call__(signal)

      Input interface operation.

      Input:
          A FloatArray of values in the range [0,1].
      Output:
          A SpikeArray of the same shape as the input.



