spark.nn.neurons.module
=======================

.. py:module:: spark.nn.neurons.module


Attributes
----------

.. autoapisummary::

   spark.nn.neurons.module.ConfigT


Classes
-------

.. autoapisummary::

   spark.nn.neurons.module.NeuronOutput
   spark.nn.neurons.module.NeuronModuleConfig
   spark.nn.neurons.module.NeuronModule


Module Contents
---------------

.. py:class:: NeuronOutput

   Bases: :py:obj:`TypedDict`


   Output ports of a neuron model.

   .. attribute:: out_spikes

      Spikes emitted by the pool on this step.

      :type: SpikeArray

   .. attribute:: Initialize self.  See help(type(self)) for accurate signature.




   .. py:attribute:: out_spikes
      :type:  spark.core.payloads.SpikeArray


.. py:class:: NeuronModuleConfig

   Bases: :py:obj:`spark.core.config.DefaultSparkConfig`


   Base configuration for neuron models written as a single module.

   :param units: Shape of the pool of neurons.
   :type units: tuple of int
   :param seed: Seed for internal random draws. Drawn from the operating system when omitted.
   :type seed: int, optional
   :param dtype: Dtype used for the internal state.
   :type dtype: DTypeLike, default jnp.float16
   :param dt: Integration step, in ms.
   :type dt: float, default 1.0


   .. py:attribute:: units
      :type:  tuple[int, ...]


   .. py:method:: __post_init__()


.. py:data:: ConfigT

.. py:class:: NeuronModule(config = None, **kwargs)

   Bases: :py:obj:`spark.core.module.SparkModule`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ ]


   Base class for neuron models written as a single module.

   A neuron assembled in Python rather than declared as a wiring of components. The
   components are held as plain attributes and stepped by `__call__`, which is the fallback
   for a model that a `Neuron` controller cannot express.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: NeuronModuleConfig, optional

   :Input Ports: **in_spikes** (*SpikeArray*) -- Spikes arriving at the pool.

   :Output Ports: **out_spikes** (*SpikeArray*) -- Spikes emitted by the pool on this step.

   .. rubric:: Notes

   Use this neuron model only when looking for specific information flows that may not be
   implemented with the default neuron controller.

   .. seealso::

      :py:obj:`Neuron`
          Controller-based neuron, declared as a wiring of components.


   .. py:attribute:: config
      :type:  ConfigT


   .. py:attribute:: units
      :value: ()



   .. py:method:: reset()

      Resets neuron states to their initial values.



   .. py:method:: __call__(in_spikes)
      :abstractmethod:


      Advances the neuron one step.

      :param in_spikes: Spikes arriving at the pool.
      :type in_spikes: SpikeArray

      :returns: Dictionary with one entry, ``out_spikes``, the spikes emitted by the pool.
      :rtype: NeuronOutput



