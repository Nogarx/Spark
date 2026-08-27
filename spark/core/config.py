#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import os
import re
import abc
import jax
import copy
import lzma
import json
import inspect
import logging
import warnings
import numpy as np
import typing as tp
import pathlib as pl
import jax.numpy as jnp
import dataclasses as dc
import spark.core.utils as utils

from math import prod
from functools import partial, wraps
from jax.typing import DTypeLike, ArrayLike
from spark.core.validation import _is_config_instance
from spark.core.registry import REGISTRY, RegistryNamespace, register_config
from spark.core.signature_parser import normalize_typehint, is_instance
from spark.core.config_validation import TypeValidator, PositiveValidator

logger = logging.getLogger('Spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class AnnotationWarning(Warning):
	"""
		Raised when the annotation of a field cannot be resolved.

		A field whose annotation cannot be read is left unchecked rather than refused, so a
		configuration still builds.
	"""
	pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

NESTED_DELIMITER = '__'
"""
	Separator addressing something inside a configuration: a nested configuration ("kernel__scale"), or one
	module of a "modules_specs" list, by its name ("synapses__kernel__scale").
"""

SHARED_DELIMITER = '_s_'
"""
	Prefix marking an argument that is handed down to every configuration below ("_s_units").
"""

_MODULE_SPECS_ANNOTATION = re.compile(r'\b(?:list|tuple|set|frozenset|Sequence|Iterable|Collection)\s*\[\s*([\w\.]+)')
"""
	Reads the element of a collection annotation, whatever container and import alias it was written with.
"""

_COLLECTION_ANNOTATION = re.compile(r'^\s*(?:[\w\.]+\.)?(list|tuple|set|frozenset|Sequence|Iterable|Collection)\b')
"""
	Recognizes an annotation that promises a collection, whatever it is a collection of.
"""

_COLLECTION_NAMES = frozenset({'list', 'tuple', 'set', 'frozenset', 'Sequence', 'Iterable', 'Collection'})

VALIDATE_CONFIGS = True
"""
	Whether a configuration checks its values against the validators its fields declare.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def validation_enabled() -> bool:
	"""
		Whether the validators of a field are being run.

		Returns
		-------
		bool
	"""
	return VALIDATE_CONFIGS

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def set_validation(enabled: bool) -> bool:
	"""
		Turns the validators on or off.

		Parameters
		----------
		enabled : bool
			True to run the validators of every field.

		Returns
		-------
		bool
			The previous setting, for restoring it afterwards.

		See Also
		--------
		NoValidation : Context manager doing the same for one block.
	"""
	global VALIDATE_CONFIGS
	previous = VALIDATE_CONFIGS
	VALIDATE_CONFIGS = bool(enabled)
	return previous

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def holds_a_collection(field: dc.Field) -> bool:
	"""
		Whether the annotation of a field says it holds a collection.

		Parameters
		----------
		field : dataclasses.Field
			Field to read.

		Returns
		-------
		bool
	"""
	annotation = field.type
	if isinstance(annotation, str):
		return _COLLECTION_ANNOTATION.match(annotation) is not None
	origin = tp.get_origin(annotation) or annotation
	return getattr(origin, '__name__', None) in _COLLECTION_NAMES

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_module_specs_field(field: dc.Field, value: tp.Any = None) -> bool:
	"""
		Whether a field holds a collection of module specifications.

		Parameters
		----------
		field : dataclasses.Field
			Field to read.
		value : Any, optional
			What the field is about to hold. Read when the annotation says nothing.

		Returns
		-------
		bool
	"""
	from spark.core.specs import ModuleSpecs
	annotation = field.type
	if isinstance(annotation, str):
		# Postponed annotation.
		match = _MODULE_SPECS_ANNOTATION.search(annotation)
		if match is not None and match.group(1).split('.')[-1] == ModuleSpecs.__name__:
			return True
	elif tp.get_origin(annotation) is not None:
		for arg in tp.get_args(annotation):
			# NOTE: A typehint can either by a class (type) or a reference that was never resolved (str)
			name = getattr(arg, '__forward_arg__', None) or getattr(arg, '__name__', None)
			if isinstance(name, str) and name.split('.')[-1] == ModuleSpecs.__name__:
				return True
	if isinstance(value, (list, tuple)) and len(value) > 0:
		if all(isinstance(v, ModuleSpecs) for v in value):
			return True
		# A configuration that came back from a file carries its specifications as dictionaries.
		if all(isinstance(v, dict) and {'name', 'module_cls'} <= v.keys() for v in value):
			return True
	return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def unflatten_kwargs(kwargs: dict[str, tp.Any], __nested_delimiter__: str = NESTED_DELIMITER, __shared_delimiter__: str = SHARED_DELIMITER) -> dict[str, tp.Any]:
	
	def _unflatten_kwargs_recursive(kwargs: dict[str, tp.Any], shared_args: dict[str, tp.Any]) -> dict[str, tp.Any]:
		unflatten_dict = {k:v for k,v in shared_args.items()}
		# Set simple arguments and discover nested kwargs
		nested_dicts = set()
		for key, value in kwargs.items():
			if __nested_delimiter__ in key and not key.startswith(__nested_delimiter__):
				nested_dicts.add(key.split(__nested_delimiter__)[0]) 
			else:
				unflatten_dict[key] = kwargs[key]
		# Unflatten nested kwargs
		for nested_key in nested_dicts:
			nested_prefix = nested_key + __nested_delimiter__
			nested_dict = {}
			# Gather associated values
			for key, value in kwargs.items():
				if key.startswith(nested_prefix):
					nested_dict[key[len(nested_prefix):]] = value
			# Unflatten dict
			unflatten_dict[nested_key] = _unflatten_kwargs_recursive(nested_dict, shared_args)
		return unflatten_dict

	# Extract shared arguments
	shared_kwargs = {}
	nested_kwargs = {}
	for key, value in kwargs.items():
		# Check if parameter is shared
		if key.startswith(__shared_delimiter__):
			shared_kwargs[key[len(__shared_delimiter__):]] = value
		else:
			nested_kwargs[key] = value

	return _unflatten_kwargs_recursive(nested_kwargs, shared_kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# NOTE: Pytree does not work with mutable values.
class StaticValue:
	"""
		Wrapper marking a value as static.

		A value wrapped this way is kept out of the traced state, so it can be read at trace time.
	"""

	__slots__ = ('value',)

	def __init__(self, value: tp.Any) -> None:
		self.value = value

	def __call__(self, **kwargs) -> tp.Any:
		return self.value

	def __array__(self, dtype=None, copy=None) -> np.ndarray:
		array = np.asarray(self.value, dtype=dtype)
		return array.copy() if copy else array

	def __repr__(self) -> str:
		return repr(self.value)

	def __len__(self) -> int:
		return len(self.value)

	def __getitem__(self, key) -> tp.Any:
		return self.value[key]

	@property
	def shape(self) -> tuple[int, ...]:
		return np.shape(self.value)

	@property
	def dtype(self):
		return np.asarray(self.value).dtype

	@staticmethod
	def unwrap(value: tp.Any) -> tp.Any:
		return value.value if isinstance(value, StaticValue) else value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class _InitNamespace:
	"""
		Proxy giving access to the initializers of a configuration.

		Reached as ``config.init.<field>``. Reading a field answers with something callable: an
		initializer when the field holds one, and a function returning the value otherwise. A
		module can therefore call ``config.init.tau(...)`` without checking which of the two it
		was given.
	"""

	def __init__(self, instance) -> None:
		self._instance = instance

	# NOTE: Partial autocomplete for dynamic interpreters
	def __dir__(self) -> list[str]:
		return [f.name for f in dc.fields(self._instance)]

	def __getattr__(self, attr_name: str) -> tp.Callable[..., ArrayLike]:

		# Validate attr_name
		if not hasattr(self._instance, attr_name):
			raise AttributeError(
				f'{type(self._instance).__name__} has no attribute "{attr_name}".'
			)
		# Get attribute
		raw_attribute = getattr(self._instance, attr_name)
		# Generate callable method
		def field_init(**kwargs) -> ArrayLike:
			from spark.nn.initializers import Initializer, InitializerConfig 
			if isinstance(raw_attribute, InitializerConfig):
				# Filter intializer kwargs
				valid_config_fields = [f.name for f in dc.fields(raw_attribute)]
				init_config_kwargs = raw_attribute.to_dict() | {k:v for k,v in kwargs.items() if k in valid_config_fields}
				# Create initializer
				initializer = raw_attribute.class_ref(**init_config_kwargs)
				# Filter call kwargs
				valid_init_kwargs = [k for k in inspect.signature(initializer).parameters]
				init_call_kwargs = {k:v for k,v in kwargs.items() if k in valid_init_kwargs}
				# Execute method
				return initializer(**init_call_kwargs)
			elif isinstance(raw_attribute, Initializer):
				# Filter call kwargs
				valid_init_kwargs = [k for k in inspect.signature(raw_attribute).parameters]
				init_call_kwargs = {k:v for k,v in kwargs.items() if k in valid_init_kwargs}
				return raw_attribute(**init_call_kwargs)
			elif isinstance(raw_attribute, SparkConfig):
				return raw_attribute.merge(**kwargs)
			elif callable(raw_attribute):
				# Filter call kwargs
				valid_fn_kwargs = [k for k in inspect.signature(raw_attribute).parameters]
				fn_call_kwargs = {k:v for k,v in kwargs.items() if k in valid_fn_kwargs}
				# Execute method
				return raw_attribute(**fn_call_kwargs)
			else:
				# Method is a simple instance
				return raw_attribute
		# NOTE: Wrap method for partial autocomplete for dynamic interpreters
		if callable(raw_attribute):
			field_init = wraps(raw_attribute)(field_init)

		return field_init

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _resolved_valid_types(cls: type) -> dict[str, tuple]:
	"""
		The types every field of a configuration class accepts, as classes rather than as strings.

		Parameters
		----------
		cls : type
			Configuration class.

		Returns
		-------
		dict of str to tuple
			Types by field name. A field whose annotation could not be resolved is absent.
	"""
	resolved = cls.__dict__.get('__resolved_valid_types__', None)
	if resolved is not None:
		return resolved
	resolved = {}
	try:
		hints = tp.get_type_hints(cls)
	except Exception as error:
		logger.debug(f'The annotations of "{cls.__name__}" could not be resolved: {error}')
		hints = {}
	for name, hint in hints.items():
		try:
			types = normalize_typehint(hint)
		except Exception:
			continue
		if types and not any(isinstance(t, str) for t in types):
			resolved[name] = types
	setattr(cls, '__resolved_valid_types__', resolved)
	return resolved

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _holds_a_tracer(value: tp.Any) -> bool:
	"""
		Whether a value is a jax tracer, or holds one.

		Parameters
		----------
		value : Any
			Value to check.

		Returns
		-------
		bool
	"""
	if isinstance(value, jax.core.Tracer):
		return True
	if isinstance(value, (tuple, list, set, frozenset)):
		return any(isinstance(entry, jax.core.Tracer) for entry in value)
	return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _validate_fields(cls: type, values: dict[str, tp.Any]) -> None:
	"""
		Runs the validators every field declares against the values it is about to hold.

		Parameters
		----------
		cls : type
			Configuration class being built.
		values : dict
			Values by field name.

		Raises
		------
		Exception
			Whatever a validator raises. Nothing is run while validation is off.
	"""
	if not validation_enabled():
		return
	resolved = _resolved_valid_types(cls)
	for field in dc.fields(cls):
		validators = field.metadata.get('validators') or ()
		if not validators:
			continue
		value = StaticValue.unwrap(values.get(field.name, None))
		if value is None:
			continue
		# NOTE: A configuration built while a function is being traced cannot be validated.
		if _holds_a_tracer(value):
			continue
		# TODO: An initializer holds an abstract representation of an array and cannot be validated.
		if field.metadata.get('allows_init', False):
			from spark.nn.initializers import Initializer, InitializerConfig
			if isinstance(value, (Initializer, InitializerConfig)):
				continue
		for validator_cls in validators:
			try:
				validator = validator_cls(field, valid_types=resolved.get(field.name, None))
			except TypeError:
				validator = validator_cls(field)
			validator.validate(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkConfigMeta(abc.ABCMeta):
	"""
		Metaclass for `SparkConfig`.

		Turns a configuration class into a dataclass, promotes every annotated attribute into a
		field, and records the parsed annotation under the ``valid_types`` metadata entry. Mutable
		defaults are rewritten as factories.
	"""

	METADATA_TEMPLATE = {
		'units': None, 
		'valid_types': None, 
		'validators': None, 
		'description': None,
		'allows_init': False,
	}

	def __new__(cls, name: str, bases: tuple[type, ...], dct: dict[str, tp.Any]) -> 'SparkConfigMeta':

		# NOTE: Every non field is promoted to field to simplify the logic of configuration objects and add metadata.
		# Iterate over annotations.
		annotations: dict[str, tp.Any] = dct.get('__annotations__', {})
		for attr_name, attr_type in annotations.items():
			# Ignore dunder methods
			if attr_name.startswith('__'):
				continue
			# Parse valid types.
			attr_typehints = cls._valid_types(attr_type)
			valid_types = {'valid_types': attr_typehints}
			allows_init = False
			for attr_type in attr_typehints:
				if not isinstance(attr_type, str):
					attr_type = str(attr_type)
				if 'Initializer' in attr_type or 'jax.Array' in attr_type or 'PlasticityParamLike' in attr_type:
					allows_init = True

			# Get value
			attr_value = dct.get(attr_name, dc.MISSING)
			default, default_factory = cls._get_default_and_factory(attr_value, attr_typehints)
			declared_metadata = dict(attr_value.metadata) if isinstance(attr_value, dc.Field) else {}
			# Construct field
			field = dc.field(
				default=default,
				default_factory=default_factory,
				metadata={
					**SparkConfigMeta.METADATA_TEMPLATE, 
					**declared_metadata,
					# NOTE: These two are read off the annotation, which is the authority on them.
					**valid_types,
					**{'allows_init': allows_init}
				}
			)
			# Set field
			setattr(cls, attr_name, field)
			dct[attr_name] = field

		# Return a warning for unannotated attributes
		for attr_name, attr_type in dct.items():
			# Ignore dunder methods
			if attr_name.startswith('__'):
				continue
			if not attr_name in annotations.keys() and not isinstance(dct[attr_name], (tp.Callable, property, classmethod)):
				warnings.warn(
					f'Attributed "{attr_name}" in configuration class "{name}" is missing an annotation, this is likely to produce errors since '
					f'Spark relies on this annotations for further processing. Please consider adding an annotation to this attribute.',
					category=AnnotationWarning,
				)

		# Update the class definition
		cls = super().__new__(cls, name, bases, dct)
		# Transform class into a dataclass
		cls = dc.dataclass(cls, kw_only=True, eq=False)

		# Wrap __init__ call to dynamically filter out invalid elements
		init_method = getattr(cls, '__init__', None)
		@wraps(init_method)
		def wrapped_init(self, **kwargs) -> None:
			from spark.core.specs import ModuleSpecs
			# Parse kwargs
			raw_shared = {k:v for k,v in kwargs.items() if k.startswith('_s_')}
			clean_shared_kwargs = {k[len('_s_'):]:v for k,v in kwargs.items() if k.startswith('_s_')}
			_kwargs = unflatten_kwargs(kwargs)
			plain_shared = {k[len(SHARED_DELIMITER):]:v for k,v in raw_shared.items()}
			clean_kwargs = {}
			# Filter invalid fields
			for field in dc.fields(cls):
				key = field.name

				# Check for module specs
				default_specs_list = _kwargs.get(field.name, None)
				# NOTE: Reaching for the default of every field would call factories that stand for nested 
				# configurations and instantiating one without proper arguments is likely to fail.
				if default_specs_list is None and holds_a_collection(field):
					if not field.default is dc.MISSING:
						default_specs_list = field.default
					elif not field.default_factory is dc.MISSING:
						default_specs_list = field.default_factory()
				if is_module_specs_field(field, default_specs_list):
					module_specs_list = []
					for module_spec in (default_specs_list or ()):
						module_spec = copy.deepcopy(module_spec)
						if isinstance(module_spec, dict):
							module_spec = ModuleSpecs.from_dict(module_spec)
						prefix = f'{module_spec.name}{NESTED_DELIMITER}'
						module_kwargs = {k[len(prefix):]:v for k,v in kwargs.items() if k.startswith(prefix)}
						spec_kwargs = raw_shared | plain_shared | module_kwargs
						# Update spec config
						if dc.is_dataclass(module_spec.config):
							module_spec.config = module_spec.config.merge(**spec_kwargs)
						elif callable(module_spec.config):
							module_spec.config = module_spec.config(**spec_kwargs)
						module_specs_list.append(module_spec)
					# Update spec list
					clean_kwargs[key] = module_specs_list
					continue

				# Common fields
				if key in _kwargs.keys():
					value = _kwargs[key]
					# Attribute is a config, forward kwargs and rebuild it
					if dc.is_dataclass(field.default) and (isinstance(value, dict) or dc.is_dataclass(value)) :
						value_dict = dc.asdict(value) if dc.is_dataclass(value) else value
						value_dict = {k:v for k,v in value_dict.items() if not v is None}
						value_cls = type(value) if dc.is_dataclass(value) and not isinstance(value, type) else type(field.default)
						base = dc.asdict(field.default) if value_cls is type(field.default) else {}
						clean_kwargs[key] = value_cls(**(base | raw_shared | plain_shared | value_dict))
					# Attribute defines factory, forward kwargs and rebuild it
					elif (not field.default_factory is dc.MISSING) and (isinstance(value, dict) or dc.is_dataclass(value)):
						if dc.is_dataclass(value) and not isinstance(value, type):
							own_fields = {f.name: getattr(value, f.name) for f in dc.fields(value)}
							clean_kwargs[key] = type(value)(**(own_fields | raw_shared | plain_shared))
							continue
						value_dict = dc.asdict(value) if dc.is_dataclass(value) else value
						value_dict = {k:v for k,v in value_dict.items() if not v is None}
						try:
							# Is this a Config factory?
							clean_kwargs[key] = field.default_factory(**(raw_shared | plain_shared | value_dict))
						except:
							# Or a simple factory? ¯\_(ツ)_/¯
							valid_fn_kwargs = [k for k in inspect.signature(field.default_factory).parameters]
							fn_call_kwargs = {k:v for k,v in value_dict.items() if k in valid_fn_kwargs}
							clean_kwargs[key] = field.default_factory(**fn_call_kwargs)
					else:
						# Parameter is a simple attribute
						if key in clean_shared_kwargs and value is None:
							clean_kwargs[key] = clean_shared_kwargs[key]
						else:
							clean_kwargs[key] = value
				else:
					# Attribute is a config, forward kwargs and rebuild it
					if dc.is_dataclass(field.default):
						clean_kwargs[key] = type(field.default)(**(dc.asdict(field.default) | raw_shared | plain_shared))
					# Attribute defines factory, forward kwargs and rebuild it
					elif (not field.default_factory is dc.MISSING):
						try:
							# Is this a Config factory?
							clean_kwargs[key] = field.default_factory(**(raw_shared | plain_shared))
						except:
							# Or a simple factory? ¯\_(ツ)_/¯
							clean_kwargs[key] = field.default_factory()
			# Freeze mutable values
			for key, value in clean_kwargs.items():
				if isinstance(value, (list, set)):
					clean_kwargs[key] = tuple(value)
				elif isinstance(value, (dict, jax.Array, np.ndarray)):
					clean_kwargs[key] = StaticValue(copy.deepcopy(value))
			# Check the values against what each field declared it accepts.
			_validate_fields(cls, clean_kwargs)
			# Call init with clean args
			return init_method(self, **clean_kwargs)
		setattr(cls, '__init__', wrapped_init)

		# Register class as pytree
		cls = jax.tree_util.register_dataclass(cls)
		return cls

	@staticmethod
	def _valid_types(attr_type: tp.Any) -> tuple[type]:
		"""
			Method to parse annotations for the attributes
		"""
		return normalize_typehint(attr_type)
		
	def _get_default_and_factory(attr_value, attr_type) -> tuple[tp.Any, tp.Any]:
		"""
			Method to map common default mutable patterns into factories 
		"""
		# Extract default and factory
		if isinstance(attr_value, dc.Field):
			default, factory = attr_value.default, attr_value.default_factory
		elif attr_value is dc.MISSING:
			# Check if defines a config, otherwise let if fail
			try:
				idx = [dc.is_dataclass(t) for t in attr_type].index(True)
				default, factory = attr_type[idx], dc.MISSING
			except:
				default, factory = dc.MISSING, dc.MISSING
		else:
			default, factory = attr_value, dc.MISSING
		# Post process factories to allow for any kwargs
		if not factory is dc.MISSING:
			if not dc.is_dataclass(factory):
				# Create a simple kwargs around the factory
				def _clean_factory(fn, **kwargs) -> tp.Callable[..., tp.Any]:
					parameters = inspect.signature(fn).parameters.values()
					if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
						return fn(**kwargs)
					valid_fn_kwargs = [p.name for p in parameters]
					fn_call_kwargs = {k:v for k,v in kwargs.items() if k in valid_fn_kwargs}
					return fn(**fn_call_kwargs)
				factory = lambda fn=factory, **kwargs: _clean_factory(fn, **kwargs)
		# Dtypes
		if utils.is_dtype(default):
			pass
		# Classes
		elif isinstance(default, type):
			factory = lambda v=default, **kwargs: v(**kwargs)
			default = dc.MISSING
		# Mutable data structures
		elif isinstance(default, (list, dict, set, jax.Array, np.ndarray)):
			factory = lambda v=default, **kwargs: copy.deepcopy(v)
			default = dc.MISSING

		return default, factory
		
#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkConfig(abc.ABC, metaclass=SparkConfigMeta):
	"""
		Base class for the configuration of a module.

		A configuration is a frozen dataclass carrying the parameters of a module. It is
		serializable, so a model can be written to a file and read back without a Python
		definition, and it validates its fields as they are set.

		Notes
		-----
		Every annotated attribute becomes a field. The metadata of a field may declare
		``validators``, ``units``, a ``description`` and ``value_options``, which the editor and
		the validators read.

		A field may be given an `Initializer` in place of a value. The array is then drawn at
		build time, once the shape is known, and is reached through ``config.init.<field>``.

		Fields named ``dt`` and ``units`` are handed down by a controller to every configuration
		it contains, so a pool is sized and clocked in one place.

		See Also
		--------
		DefaultSparkConfig : Adds the seed, dtype and dt every module needs.
	"""

	@classmethod
	def partial(cls, **kwargs) -> 'SparkConfig':
		# Set None as a default value
		for field in dc.fields(cls):
			if field.default is dc.MISSING and field.default_factory is dc.MISSING and (not field.name in kwargs):
				kwargs[field.name] = None
		return cls(**kwargs)

	def merge(self, **kwargs) -> 'SparkConfig':
		"""
			Returns a copy of this configuration with the given values written over it.

			Parameters
			----------
			**kwargs
				Values by field name.

			Returns
			-------
			SparkConfig
				A new configuration. This one is left unchanged.
		"""
		_self = {field.name: getattr(self, field.name) for field in dc.fields(self)}
		return type(self)(**(_self | kwargs))

	@property
	def init(self,):
		return _InitNamespace(self,)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

	# TODO: This method is not ideal. It solves the module association problem in a very brittle way. 
	# There should be another better pattern for this problem.
	@property
	def class_ref(obj: 'SparkConfig') -> type:
		"""
			Returns the module or initializer class this configuration belongs to.

			Returns
			-------
			type
		"""
		# TODO: This could probably be handle more gracefully (part of the todo above)
		# Check if this Config is for a controller (Brain/Neuron) 
		from spark.nn.controllers.brain import Brain, BrainConfig
		from spark.nn.controllers.neuron import Neuron, NeuronConfig
		if isinstance(obj, BrainConfig):
			return Brain
		elif is_instance(obj, NeuronConfig):
			return Neuron
		# Check for class_ref otherwise try to set it up.
		if getattr(obj, '__class_ref__', None) is None:
			if obj.__class__.__name__[-6:].lower() == 'config':
				obj.__class_ref__ = obj.__class__.__name__[:-6]
			else:
				# Config is not following convention, manual input of __class_ref__ is required.
				raise AttributeError(
					f'Configuration \"{obj.__name__}\" does not define a __class_ref__.'
				)
		# Currently it can only be either a Module or a Initializer, so better check those two.
		module_class_ref = REGISTRY.Components.get(obj.__class_ref__)
		initializer_class_ref =REGISTRY.Initializers.get(obj.__class_ref__)
		interface_class_ref = REGISTRY.Interfaces.get(obj.__class_ref__)
		# Check we only got one coincidence, otherwise throw an error to avoid headaches.
		if module_class_ref and initializer_class_ref or module_class_ref and interface_class_ref:
			raise AttributeError(
				f'Configuration \"{obj.__class__.__name__}\" cannot resolve __class_ref__. '
				f'A Module and an Initializer with the same reference were found. '
				f'To prevent errors impute the class manually. Alternatively, update the name '
				f'of one of the classes to avoid overlappings.'
			)
		if module_class_ref:
			class_ref = module_class_ref.get_cls()
		elif initializer_class_ref: 
			class_ref = initializer_class_ref.get_cls()
		elif interface_class_ref: 
			class_ref = interface_class_ref.get_cls()
		else:
			raise AttributeError(
				f'Configuration \"{obj.__class__.__name__}\" cannot resolve __class_ref__. '
				f'No Module nor Initializer with the same reference were found. '
				f'Either rename the configuration object as \"Object.__class__.__name__ + Config\" or'
				f'manually define __class_ref__ using the registry name of the object (default: Object.__class__.__name__).'
			)
		return class_ref



	def __iter__(self) -> tp.Iterator[tuple[str, tp.Any]]:
		"""
			Iterates over the fields of the configuration.

			Yields
			------
			field_name : str
				Name of the field.
			field_value : Any
				Value the field holds.
		"""
		# Iterate over all defined fields of the dataclass
		for f in dc.fields(self):
			if f.name.startswith('__'):
				continue
			# Yield the field name and its corresponding value
			value = getattr(self, f.name, None)
			yield (f.name, value)



	def inspect(self, simplified=False) -> str:
		"""
			Prints the tree of fields of this configuration.
		"""
		print(utils.ascii_tree(self._parse_tree_structure(0, simplified=simplified)))



	def _inspect(self, simplified=True) -> str:
		"""
			Returns a formated string of the datastructure.
		"""
		return utils.ascii_tree(self._parse_tree_structure(0, simplified=simplified))



	def _parse_tree_structure(self, current_depth: int, simplified: bool = False, header: str | None= None) -> str:
		"""
			Parses the tree to produce a string with the appropiate format for the ascii_tree method.
		"""
		from spark.core.specs import ModuleSpecs
		level_header = f'{header}: ' if header else ''
		rep = current_depth * ' ' + f'{level_header}{self.__class__.__name__}\n'
		for field in dc.fields(self):
			name = field.name
			value = getattr(self, field.name, None)
			if not simplified:
				if isinstance(value, SparkConfig):
					rep += value._parse_tree_structure(current_depth+1, simplified=simplified, header=name)
				else:
					# Module spec lists
					if isinstance(value, (list, tuple)) and len(value) > 0 and all([isinstance(v, ModuleSpecs) for v in value]):
						rep += (current_depth+1) * ' ' + f'{name}: tuple[ModuleSpecs, ...]\n'
						for spec in value:
							rep += spec.config._parse_tree_structure(current_depth+2, simplified=simplified, header=spec.name)
						continue
					# Iterables
					if isinstance(value, (list, tuple, set)) and len(value) > 5:
						value_str = str(value_str[:5])
						value_str = f'{type(value)}([{value_str[1:-1]}, ...])'
					elif isinstance(value, (np.ndarray, jnp.ndarray)) and prod(value.shape) > 5:
						value_str = ', '.join([f'{x:.2f}'.rstrip('0').rstrip('.') for x in value.reshape(-1)[:5]]).strip('\n').replace('\n', '')
						value_str = f'array([{value_str[:-1]}, ...], dtype={value.dtype})'
					else:
						value_str = str(value).strip('\n')
					# Missing types
					if field.type == jax.typing.DTypeLike:
						field_types = 'DTypeLike'
					elif isinstance(value, (np.ndarray, jnp.ndarray)):
						field_types = 'ArrayLike'
					elif isinstance(field.type, type):
						field_types = field.type.__name__
					else:
						field_types = str(type(value).__name__)
					rep += (current_depth+1) * ' ' + f'{name}: {field_types} <- {value_str}\n'
			else:
				if isinstance(value, SparkConfig):
					rep += value._parse_tree_structure(current_depth+1, simplified=simplified)
				elif isinstance(value, (list, tuple)) and all([isinstance(v, ModuleSpecs) for v in value]):
					rep += (current_depth+1) * ' ' + f'ModuleSpecs\n'
					for spec in value:
						rep += spec.config._parse_tree_structure(current_depth+2, simplified=simplified)
		return rep



	def with_new_seeds(self, seed: int | None = None) -> 'SparkConfig':
		"""
			Returns a copy of this configuration with every seed redrawn.

			Returns
			-------
			SparkConfig
				A new configuration. This one is left unchanged.
		"""
		from spark.core.specs import ModuleSpecs

		def _with_new_seeds(config: 'SparkConfig', _seed: int) -> 'SparkConfig':
			# Current config to dict
			_config = copy.deepcopy(config)
			# Jax key
			key = jax.random.key(_seed)
			for field in dc.fields(config):
				if dc.is_dataclass(getattr(_config, field.name, None)):
					# Create new seed
					key, subkey = jax.random.split(key, 2)
					new_seed = int(subkey._base_array[0])
					# Rebuild nested config with new seed
					setattr(_config, field.name, _with_new_seeds(getattr(_config, field.name), new_seed)) 
				elif is_module_specs_field(field, getattr(_config, field.name, None)):
					module_specs_list = []
					for module_spec in getattr(_config, field.name, []):
						module_spec: ModuleSpecs = copy.deepcopy(module_spec)
						# Create new seed
						key, subkey = jax.random.split(key, 2)
						new_seed = int(subkey._base_array[0])
						# Update spec config
						module_spec.config = _with_new_seeds(module_spec.config, new_seed)
						module_specs_list.append(module_spec)
					# Update spec list
					setattr(_config, field.name, module_specs_list)
				elif field.name == 'seed':
					# Update config seed
					key, subkey = jax.random.split(key, 2)
					new_seed = int(subkey._base_array[0])
					setattr(_config, 'seed', new_seed)
			# Rebuild current config
			return _config

		# Generate a new seed if none was provided
		seed = int.from_bytes(os.urandom(4), 'little') if seed is None else seed
		return _with_new_seeds(self, seed)



	def to_dict(self,) -> dict[str, dict[str, tp.Any]]:
		"""
			Serializes the configuration to a dictionary.

			Returns
			-------
			dict
		"""

		def _clean_value(value: tp.Any):
			value = StaticValue.unwrap(value)
			if isinstance(value, dict):
				return _clean_dict(value)
			if isinstance(value, (list, tuple)):
				return type(value)(_clean_value(v) for v in value)
			return value

		def _clean_dict(dct: dict[str, tp.Any]):
			for k in list(dct.keys()):
				if k.startswith('__'):
					dct.pop(k)
				else:
					dct[k] = _clean_value(dct[k])
			return dct

		return _clean_dict(dc.asdict(self))



	@classmethod
	def from_dict(cls: type['SparkConfig'], dct: dict[str, tp.Any]) -> 'SparkConfig':
		"""
			Builds a configuration from a dictionary.

			Parameters
			----------
			dct : dict
				As produced by `to_dict`.

			Returns
			-------
			SparkConfig
		"""
		return cls(**dct)



	def to_file(
			self,
			file_path: str,
			compress: bool = True,
			verbose: bool = True,
			metadata: dict[str, tp.Any] | None = None,
		) -> None:
		"""
			Writes the configuration to a .scfg file.

			Parameters
			----------
			file_path : str
				Where to write.
			compress : bool, default True
				Compress the file.
			verbose : bool, default True
				Log where the file was written.
			metadata : dict, optional
				Written beside the configuration. `from_file` does not read it back; use
				`metadata_from_file` for that. The editor stores node positions here.
		"""
		# Validate path
		path = pl.Path(file_path)
		# Ensure the parent directory exists.
		path.parent.mkdir(parents=True, exist_ok=True)
		# Write to file.
		from spark.core.serializer import SparkJSONEncoder
		reg = REGISTRY.Configs.get_by_cls(self.__class__)
		if not reg:
			raise RuntimeError(
				f'Config class "{self.__class__}" is not in the registry.'
				f'Reconstruction from unregistered classes is not currently possible.'
				f'Use the "register_config" decorator to add the class to the registry.'
			)
		payload = json.dumps(self, cls=SparkJSONEncoder, indent=4, metadata=metadata)
		opener = lzma.open if compress else open
		mode = 'wt' if compress else 'w'
		temp_path = path.with_name(f'{path.name}.partial')
		try:
			with opener(temp_path, mode, encoding='utf-8') as json_file:
				json_file.write(payload)
			os.replace(temp_path, path)
		finally:
			temp_path.unlink(missing_ok=True)
		if verbose:
			print(f'Configuration saved to "{path}".')



	@classmethod
	def metadata_from_file(cls, file_path: str) -> dict[str, tp.Any]:
		"""
			Reads the metadata written beside the configuration of a file.

			The configuration itself is not decoded, so this works for a file naming models that are
			not registered.

			Parameters
			----------
			file_path : str
				File to read.

			Returns
			-------
			dict
				What the writer stored, empty when it stored nothing.
		"""
		from spark.core.serializer import METADATA_KEY
		path = pl.Path(file_path)
		if not path.is_file():
			raise FileNotFoundError(f'No file found at the specified path: "{path}".')
		with open(path, 'rb') as f:
			is_compressed = (f.read(6) == b'\xfd7zXZ\x00')
		opener = lzma.open if is_compressed else open
		mode = 'rt' if is_compressed else 'r'
		with opener(path, mode, encoding='utf-8') as json_file:
			document = json.load(json_file)
		metadata = document.get(METADATA_KEY) if isinstance(document, dict) else None
		return metadata if isinstance(metadata, dict) else {}

	@classmethod
	def from_file(cls: type['SparkConfig'], file_path: str) -> 'SparkConfig':
		"""
			Builds a configuration from a .scfg file.

			Parameters
			----------
			file_path : str
				File to read.

			Returns
			-------
			SparkConfig
		"""
		path = pl.Path(file_path)
		# Validate path
		if not path.is_file():
			raise FileNotFoundError(f'No file found at the specified path: "{path}".')
		# Read the header to determine if the file start with the magic bytes: \xfd7zXZ\x00 (is LZMA compressed)
		with open(path, 'rb') as f:
			magic_bytes = f.read(6)
		is_compressed = (magic_bytes == b'\xfd7zXZ\x00')
		# Parse the file
		opener = lzma.open if is_compressed else open
		mode = 'rt' if is_compressed else 'r'
		with opener(path, mode, encoding='utf-8') as json_file:
			payload = json_file.read()
		from spark.core.serializer import SparkJSONDecoder
		from spark.core.registry import register_models_from_payload
		register_models_from_payload(json.loads(payload))
		obj = json.loads(payload, cls=SparkJSONDecoder)
		if not _is_config_instance(obj):
			raise TypeError(
				f'Expected final object to be of type "SparkConfig" but after decoding the final object was of type "{obj.__class__}".'
			)
		return obj
		
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class DefaultSparkConfig(SparkConfig):
    """
        Configuration of a module, with the fields every module needs.

        Parameters
        ----------
        seed : int, optional
            Seed for the random draws of the module. Drawn from the operating system when omitted.
        dtype : DTypeLike, default jnp.float16
            Dtype used for the internal state.
        dt : float, default 1.0
            Integration step, in ms.
    """
    seed: int = dc.field(
        default_factory=lambda: int.from_bytes(os.urandom(4), 'little'), 
        metadata={
            'validators': [
                TypeValidator,
            ], 
            'description': 'Seed for internal random processes.',
        })
    dtype: DTypeLike = dc.field(
        default=jnp.float16, 
        metadata={
            'validators': [
                TypeValidator,
            ], 
            'value_options': [
                jnp.float16,
                jnp.float32,
            ],
            'description': 'Dtype used for JAX dtype promotions.',
        })
    dt: float = dc.field(
        default=1.0, 
        metadata={
            'units': 'ms',
            'validators': [
                TypeValidator,
                PositiveValidator,
            ],
            'description': 'Deltatime integration constant.',
        })
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################