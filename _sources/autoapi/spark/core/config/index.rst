spark.core.config
=================

.. py:module:: spark.core.config


Attributes
----------

.. autoapisummary::

   spark.core.config.logger
   spark.core.config.NESTED_DELIMITER
   spark.core.config.SHARED_DELIMITER
   spark.core.config.CONFIG_EXTENSION
   spark.core.config.VALIDATE_CONFIGS


Exceptions
----------

.. autoapisummary::

   spark.core.config.AnnotationWarning


Classes
-------

.. autoapisummary::

   spark.core.config.StaticValue
   spark.core.config.SparkConfigMeta
   spark.core.config.SparkConfig
   spark.core.config.DefaultSparkConfig


Functions
---------

.. autoapisummary::

   spark.core.config.validation_enabled
   spark.core.config.set_validation
   spark.core.config.holds_a_collection
   spark.core.config.is_module_specs_field
   spark.core.config.unflatten_kwargs


Module Contents
---------------

.. py:data:: logger

.. py:exception:: AnnotationWarning

   Bases: :py:obj:`Warning`


   Raised when the annotation of a field cannot be resolved.

   A field whose annotation cannot be read is left unchecked rather than refused, so a
   configuration still builds.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:data:: NESTED_DELIMITER
   :value: '__'


   a nested configuration ("kernel__scale"), or one
   module of a "modules_specs" list, by its name ("synapses__kernel__scale"). It is the dot of an address
   ("synapses.kernel.scale") written in a keyword argument (see `spark.core.addresses`).

   :type: Separator addressing something inside a configuration

.. py:data:: SHARED_DELIMITER
   :value: '_s_'


   Prefix marking an argument that is handed down to every configuration below ("_s_units"), as the pattern
   "**.units" would (see `spark.core.addresses`).

.. py:data:: CONFIG_EXTENSION
   :value: '.scfg'


   Extension of the files written by `SparkConfig.to_file`.

.. py:data:: VALIDATE_CONFIGS
   :value: True


   Whether a configuration checks its values against the validators its fields declare.

.. py:function:: validation_enabled()

   Whether the validators of a field are being run.

   :rtype: bool


.. py:function:: set_validation(enabled)

   Turns the validators on or off.

   :param enabled: True to run the validators of every field.
   :type enabled: bool

   :returns: The previous setting, for restoring it afterwards.
   :rtype: bool

   .. seealso::

      :py:obj:`NoValidation`
          Context manager doing the same for one block.


.. py:function:: holds_a_collection(field)

   Whether the annotation of a field says it holds a collection.

   :param field: Field to read.
   :type field: dataclasses.Field

   :rtype: bool


.. py:function:: is_module_specs_field(field, value = None)

   Whether a field holds a collection of module specifications.

   :param field: Field to read.
   :type field: dataclasses.Field
   :param value: What the field is about to hold. Read when the annotation says nothing.
   :type value: Any, optional

   :rtype: bool


.. py:function:: unflatten_kwargs(kwargs, __nested_delimiter__ = NESTED_DELIMITER, __shared_delimiter__ = SHARED_DELIMITER)

.. py:class:: StaticValue(value)

   Wrapper marking a value as static.

   A value wrapped this way is kept out of the traced state, so it can be read at trace time.


   .. py:attribute:: __slots__
      :value: ('value',)



   .. py:attribute:: value


   .. py:method:: __call__(**kwargs)


   .. py:method:: __array__(dtype=None, copy=None)


   .. py:method:: __repr__()


   .. py:method:: __len__()


   .. py:method:: __getitem__(key)


   .. py:property:: shape
      :type: tuple[int, ...]



   .. py:property:: dtype


   .. py:method:: unwrap(value)
      :staticmethod:



.. py:class:: SparkConfigMeta

   Bases: :py:obj:`abc.ABCMeta`


   Metaclass for `SparkConfig`.

   Turns a configuration class into a dataclass, promotes every annotated attribute into a
   field, and records the parsed annotation under the ``valid_types`` metadata entry. Mutable
   defaults are rewritten as factories.


   .. py:attribute:: METADATA_TEMPLATE


.. py:class:: SparkConfig

   Bases: :py:obj:`abc.ABC`


   Base class for the configuration of a module.

   A configuration is a frozen dataclass carrying the parameters of a module. It is
   serializable, so a model can be written to a file and read back without a Python
   definition, and it validates its fields as they are set.

   .. rubric:: Notes

   Every annotated attribute becomes a field. The metadata of a field may declare
   ``validators``, ``units``, a ``description`` and ``value_options``, which the editor and
   the validators read.

   A field may be given an `Initializer` in place of a value. The array is then drawn at
   build time, once the shape is known, and is reached through ``config.init.<field>``.

   Fields named ``dt`` and ``units`` are handed down by a controller to every configuration
   it contains, so a pool is sized and clocked in one place.

   .. seealso::

      :py:obj:`DefaultSparkConfig`
          Adds the seed, dtype and dt every module needs.


   .. py:method:: partial(**kwargs)
      :classmethod:



   .. py:method:: merge(**kwargs)

      Returns a copy of this configuration with the given values written over it.

      :param \*\*kwargs: Values by field name.

      :returns: A new configuration. This one is left unchanged.
      :rtype: SparkConfig



   .. py:property:: init


   .. py:method:: update(changes)

      Returns a copy of this configuration with the parts at the given addresses replaced.

      :param changes: Values by address or pattern, applied in order. An address is the dotted chain of names
                      leading to a part: modules by name and nested configurations by field name, then the
                      field, as ``A_excitatory.soma`` or ``A_excitatory.soma.threshold``. A pattern matches
                      several: ``*_excitatory.soma``, ``**.threshold`` (see `spark.core.addresses`). What an
                      address names is replaced by its value:

                      * a module, by a configuration of its class, or by a `ModuleSpecs` of the same name,
                        which also sets its class and its wiring;
                      * a field, by the value, a nested configuration included.
      :type changes: mapping of str to Any

      :returns: A new configuration. This one, and the values given, are left unchanged.
      :rtype: SparkConfig

      :raises KeyError: If an address or a pattern matches nothing.
      :raises ValueError: If a pattern matches both a part and a part within it, or a `ModuleSpecs` has another
          name than the module it replaces.
      :raises TypeError: If a module is given something else than a configuration of its class or a
          `ModuleSpecs`.

      .. rubric:: Notes

      Each part matched takes a copy of its value. The controllers holding it hand it their
      ``units`` and ``dt`` again, so these are changed on the controller (``A_excitatory.units``):
      a value set below it is overwritten. A configuration given without a seed takes one derived
      from the seed of the controller holding it and its name, so that two modules given the same
      configuration still draw different values. A seed given is kept.

      .. rubric:: Examples

      >>> config = config.update({
      ...     '*_excitatory.soma': spark.nn.somas.AdaptiveLeakySomaConfig(threshold=-45.0),
      ...     '**.synapses.kernel.scale': 2000.0,
      ...     'integrator': spark.ModuleSpecs(name='integrator', module_cls=..., inputs=...),
      ... })



   .. py:method:: addresses(pattern = None)

      Returns the addresses of the parts of this configuration.

      :param pattern: Keeps the addresses it matches (see `spark.core.addresses`).
      :type pattern: str, optional

      :returns: The address of every module, nested configuration and field, depth first.
      :rtype: tuple of str



   .. py:method:: __getitem__(address)

      Returns the part of this configuration at an address.

      :param address: The dotted chain of names leading to the part, as ``A_excitatory.soma.threshold``.
      :type address: str

      :returns: The value of a field, or the configuration of a module.
      :rtype: Any

      :raises KeyError: If nothing is at the address.



   .. py:property:: class_ref
      :type: type


      Returns the module or initializer class this configuration belongs to.

      :rtype: type


   .. py:method:: __iter__()

      Iterates over the fields of the configuration.

      :Yields: * **field_name** (*str*) -- Name of the field.
               * **field_value** (*Any*) -- Value the field holds.



   .. py:method:: inspect(simplified=False)

      Prints the tree of fields of this configuration.



   .. py:method:: with_new_seeds(seed = None)

      Returns a copy of this configuration with every seed redrawn.

      :returns: A new configuration. This one is left unchanged.
      :rtype: SparkConfig



   .. py:method:: to_dict()

      Serializes the configuration to a dictionary.

      :rtype: dict



   .. py:method:: from_dict(dct)
      :classmethod:


      Builds a configuration from a dictionary.

      :param dct: As produced by `to_dict`.
      :type dct: dict

      :rtype: SparkConfig



   .. py:method:: to_file(file_path, compress = True, verbose = True, metadata = None)

      Writes the configuration to a .scfg file.

      :param file_path: Where to write. ``.scfg`` is added to the name when it does not end with it.
      :type file_path: str
      :param compress: Compress the file.
      :type compress: bool, default True
      :param verbose: Log where the file was written.
      :type verbose: bool, default True
      :param metadata: Written beside the configuration. `from_file` does not read it back; use
                       `metadata_from_file` for that. The editor stores node positions here.
      :type metadata: dict, optional

      :returns: The file written.
      :rtype: pathlib.Path



   .. py:method:: metadata_from_file(file_path)
      :classmethod:


      Reads the metadata written beside the configuration of a file.

      The configuration itself is not decoded, so this works for a file naming models that are
      not registered.

      :param file_path: File to read, with or without its ``.scfg`` extension.
      :type file_path: str

      :returns: What the writer stored, empty when it stored nothing.
      :rtype: dict



   .. py:method:: from_file(file_path)
      :classmethod:


      Builds a configuration from a .scfg file.

      :param file_path: File to read, with or without its ``.scfg`` extension.
      :type file_path: str

      :rtype: SparkConfig



.. py:class:: DefaultSparkConfig

   Bases: :py:obj:`SparkConfig`


   Configuration of a module, with the fields every module needs.

   :param seed: Seed for the random draws of the module. When omitted, derived from the seed of the
                controller holding the module and the name of the module, or drawn from the operating
                system for a module on its own.
   :type seed: int, optional
   :param dtype: Dtype used for the internal state.
   :type dtype: DTypeLike, default jnp.float16
   :param dt: Integration step, in ms.
   :type dt: float, default 1.0


   .. py:attribute:: seed
      :type:  int | None


   .. py:attribute:: dtype
      :type:  jax.typing.DTypeLike


   .. py:attribute:: dt
      :type:  float


