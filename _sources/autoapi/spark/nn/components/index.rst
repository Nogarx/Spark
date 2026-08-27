spark.nn.components
===================

.. py:module:: spark.nn.components


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/spark/nn/components/base/index
   /autoapi/spark/nn/components/delays/index
   /autoapi/spark/nn/components/plasticity/index
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


   Base class for the components a neuron is built from.

   :param config: Model configuration. Its fields may also be given as keyword arguments.
   :type config: ComponentConfig, optional

   :Input Ports: **\*\*inputs** (*SparkPayload*) -- Declared by the concrete component through the signature of its ``__call__``.

   :Output Ports: **\*\*outputs** (*SparkPayload*) -- Declared by the concrete component through the TypedDict its ``__call__`` returns.

   .. seealso::

      :py:obj:`Soma`
          Membrane potential and spike generation.

      :py:obj:`Synapses`
          Presynaptic spikes to postsynaptic current.

      :py:obj:`Delays`
          Conduction delays.

      :py:obj:`Plasticity`
          Weight updates.


   .. py:attribute:: config
      :type:  ConfigT


.. py:class:: ComponentConfig

   Bases: :py:obj:`spark.core.config.DefaultSparkConfig`


   Base configuration for components.

   :param seed: Seed for internal random draws. Drawn from the operating system when omitted.
   :type seed: int, optional
   :param dtype: Dtype used for the internal state.
   :type dtype: DTypeLike, default jnp.float16
   :param dt: Integration step, in ms. Overwritten by the enclosing controller.
   :type dt: float, default 1.0


