spark.core.module
=================

.. py:module:: spark.core.module


Attributes
----------

.. autoapisummary::

   spark.core.module.ConfigT
   spark.core.module.InputT


Classes
-------

.. autoapisummary::

   spark.core.module.ModuleOutput
   spark.core.module.SparkMeta
   spark.core.module.SparkModule


Module Contents
---------------

.. py:data:: ConfigT

.. py:data:: InputT

.. py:class:: ModuleOutput

   Bases: :py:obj:`TypedDict`


   Spark module output template

   Initialize self.  See help(type(self)) for accurate signature.


.. py:class:: SparkMeta

   Bases: :py:obj:`flax.nnx.module.ModuleMeta`


   Metaclass for Spark Modules.


.. py:class:: SparkModule(*, config = None, name = None, **kwargs)

   Bases: :py:obj:`flax.nnx.Module`, :py:obj:`abc.ABC`, :py:obj:`Generic`\ [\ :py:obj:`ConfigT`\ , :py:obj:`InputT`\ ]


   Base class for Spark Modules


   .. py:attribute:: name
      :type:  str
      :value: 'name'



   .. py:attribute:: config
      :type:  ConfigT


   .. py:attribute:: default_config
      :type:  type[ConfigT]


   .. py:method:: __init_subclass__(**kwargs)
      :classmethod:



   .. py:attribute:: __built__
      :type:  bool
      :value: False



   .. py:attribute:: __allow_cycles__
      :type:  bool
      :value: False



   .. py:method:: get_config_spec()
      :classmethod:


      Returns the default configuration class associated with this module.



   .. py:method:: build(input_specs)

      Build method.



   .. py:method:: reset()

      Reset module to its default state.



   .. py:method:: set_contract_output_specs(contract_specs)

      Recurrent shape policy pre-defines expected shapes for the output specs.

      This is function is a binding contract that allows the modules to accept self connections.

      Input:
          contract_specs: dict[str, PortSpecs], A dictionary with a contract for the output specs.

      NOTE: If both, shape and output_specs, are provided, output_specs takes preference over shape.



   .. py:method:: get_contract_output_specs()

      Retrieve the recurrent spec policy of the module.



   .. py:method:: get_input_specs()

      Returns a dictionary of the SparkModule's input port specifications.



   .. py:method:: get_output_specs()

      Returns a dictionary of the SparkModule's input port specifications.



   .. py:method:: get_rng_keys(num_keys)

      Generates a new collection of random keys for the JAX's random engine.



   .. py:method:: __call__(**kwargs)
      :abstractmethod:


      Execution method.



   .. py:method:: __repr__()


   .. py:method:: inspect()

      Returns a formated string of the datastructure.



   .. py:method:: checkpoint(path, overwrite=False)


   .. py:method:: from_checkpoint(path, safe=True)
      :classmethod:



