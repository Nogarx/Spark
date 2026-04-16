#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import os
import abc
import jax
import copy
import lzma
import json
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
from spark.core.registry import REGISTRY, register_config
from spark.core.signature_parser import normalize_typehint, is_instance
from spark.core.config_validation import TypeValidator, PositiveValidator

logger = logging.getLogger('Spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class AnnotationWarning(Warning):
	pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def unflatten_kwargs(kwargs: dict[str, tp.Any], __nested_delimiter__: str = '__', __shared_delimiter__: str = '_s_') -> dict[str, tp.Any]:
	
	def _unflatten_kwargs_recursive(kwargs: dict[str, tp.Any], shared_args: dict[str, tp.Any]) -> dict[str, tp.Any]:
		unflatten_dict = {k:v for k,v in shared_args.items()}
		# Set simple arguments and discover nested kwargs
		nested_dicts = set()
		for key, value in kwargs.items():
			if __nested_delimiter__ in key:
				nested_dicts.add(key.split(__nested_delimiter__)[0]) 
			else:
				unflatten_dict[key] = kwargs[key]
		# Unflatten nested kwargs
		for nested_key in nested_dicts:
			nested_dict = {}
			# Gather associated values
			for key, value in kwargs.items():
				if key.startswith(nested_key):
					nested_dict[key[len(nested_key+__nested_delimiter__):]] = value
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

class SparkConfigMeta(abc.ABCMeta):
	"""
		Spark Configuration Metaclass
	"""

	METADATA_TEMPLATE = {
		'units': None, 
		'valid_types': None, 
		'validators': None, 
		'description': None,
	}

	def __new__(cls, name: str, bases: tuple[type, ...], dct: dict[str, tp.Any]) -> 'SparkConfigMeta':

		# NOTE: Every non field is promoted to field to simplify the logic of configuration objects and add metadata.
		# Iterate over annotations.
		annotations: dict[str, tp.Any] = dct.get('__annotations__', {})
		for attr_name, attr_type in annotations.items():
			# Parse valid types.
			attr_typehints = cls._valid_types(attr_type)
			valid_types = {'valid_types': attr_typehints}
			# Get value
			attr_value = dct.get(attr_name, dc.MISSING)
			default, default_factory = cls._get_default_and_factory(attr_value, attr_typehints)
			# Construct field
			field = dc.field(
				default=default,
				default_factory=default_factory,
				metadata={
					**SparkConfigMeta.METADATA_TEMPLATE, 
					**valid_types,
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
			# Filter fields
			valid_fields = {f.name: f for f in dc.fields(cls)}
			_kwargs = unflatten_kwargs(kwargs)
			clean_kwargs = {}
			for key, value in _kwargs.items():
				# Remove invalid fields
				if not key in valid_fields.keys():
					continue
				field = valid_fields[key]
				# Attribute is a config, forward kwargs and rebuild it
				if dc.is_dataclass(field.default) and (isinstance(value, dict) or dc.is_dataclass(value)) :
					value_dict = dc.asdict(value) if dc.is_dataclass(value) else value
					clean_kwargs[key] = type(field.default)(**{**dc.asdict(field.default), **value_dict})
				# Attribute defines factory, forward kwargs and rebuild it
				elif (not field.default_factory is dc.MISSING) and (isinstance(value, dict) or dc.is_dataclass(value)):
					value_dict = dc.asdict(value) if dc.is_dataclass(value) else value
					clean_kwargs[key] = field.default_factory(**value_dict)
				else:
					# Parameter is a simple attribute
					clean_kwargs[key] = value
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
		if isinstance(attr_type, str):
			raise RuntimeError(f'_valid_types got an str "{attr_type}".')
		else:
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
		# Dtypes
		if utils.is_dtype(default):
			pass
		# Classes
		elif isinstance(default, type):
			factory = lambda v=default, **kwargs: v(**kwargs)
			default = dc.MISSING
		# Mutable data structures
		elif isinstance(default, (list, dict, set, ArrayLike)):
			factory = lambda v=default: copy.deepcopy(v)
			default = dc.MISSING
		return default, factory
		
#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkConfig(abc.ABC, metaclass=SparkConfigMeta):
	"""
		Base Configuration.
	"""

	@classmethod
	def partial(cls, **kwargs) -> 'SparkConfig':
		# Set None as a default value
		for field in dc.fields(cls):
			if field.default is dc.MISSING and field.default_factory is dc.MISSING and (not field.name in kwargs):
				kwargs[field.name] = None
		return cls(**kwargs)

	# TODO: This method is not ideal. It solves the module association problem in a very brittle way. 
	# There should be another better pattern for this problem.
	@property
	def class_ref(obj: 'SparkConfig') -> type:
		"""
			Returns the type of the associated Module/Initializer.

			NOTE: It is recommended to set the __class_ref__ to the name of the associated module/initializer
			when defining custom configuration classes. The automatic class_ref solver is extremely brittle and
			likely to fail in many different custom scenarios.
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
		module_class_ref = REGISTRY.MODULES.get(obj.__class_ref__)
		initializer_class_ref = REGISTRY.INITIALIZERS.get(obj.__class_ref__)
		# Check we only got one coincidence, otherwise throw an error to avoid headaches.
		if module_class_ref and initializer_class_ref:
			raise AttributeError(
				f'Configuration \"{obj.__class__.__name__}\" cannot resolve __class_ref__. '
				f'A Module and an Initializer with the same reference were found. '
				f'To prevent errors impute the class manually. Alternatively, update the name '
				f'of one of the classes to avoid overlappings.'
			)
		if module_class_ref:
			class_ref = module_class_ref.class_ref
		elif initializer_class_ref: 
			class_ref = initializer_class_ref.class_ref
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
			(key, value) iterator.

			Output:
				field_name: str, field name 
				field_value: tp.Any, field value
		"""
		# Iterate over all defined fields of the dataclass
		for f in dc.fields(self):
			# Yield the field name and its corresponding value
			value = getattr(self, f.name, None)
			yield (f.name, value)



	def inspect(self, simplified=False) -> str:
		"""
			Returns a formated string of the datastructure.
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
						rep += (current_depth+1) * ' ' + f'{name}: list[ModuleSpecs]\n'
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
			Utility method to recompute all seed variables within the SparkConfig.
			Useful when creating several populations from the same config.
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
				elif field.type is list[ModuleSpecs]:
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
			Serialize config to dictionary
		"""
		return dc.asdict(self)



	@classmethod
	def from_dict(cls: type['SparkConfig'], dct: dict[str, tp.Any]) -> 'SparkConfig':
		"""
			Create config instance from dictionary.
		"""
		return cls(**dct)



	def to_file(self, file_path: str, compress: bool = True, verbose: bool = True) -> None:
		"""
			Export a config instance from a .scfg file.
		"""
		# Validate the config
		# If partial is True, values with errors are replaced with a None.
		#self.validate(is_partial=is_partial)
		# Validate path
		path = pl.Path(file_path)
		# Ensure the parent directory exists.
		path.parent.mkdir(parents=True, exist_ok=True)
		# Write to file.
		from spark.core.serializer import SparkJSONEncoder
		opener = lzma.open if compress else open
		mode = 'wt' if compress else 'w'
		with opener(path, mode, encoding='utf-8') as json_file:
			reg = REGISTRY.CONFIG.get_by_cls(self.__class__)
			if not reg:
				raise RuntimeError(
					f'Config class "{self.__class__}" is not in the registry.'
					f'Reconstruction from unregistered classes is not currently possible.'
					f'Use the "register_config" decorator to add the class to the registry.'
				)
			# Add top config metadata
			encoder_cls = SparkJSONEncoder #partial(SparkJSONEncoder, is_partial=is_partial)
			json.dump(self, json_file, cls=encoder_cls, indent=4)
		if verbose:
			print(f'Configuration saved to "{path}".')



	@classmethod
	def from_file(cls: type['SparkConfig'], file_path: str, is_partial: bool = False) -> 'SparkConfig':
		"""
			Create config instance from a .scfg file.
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
			# Try to decode
			from spark.core.serializer import SparkJSONDecoder
			decoder_cls = partial(SparkJSONDecoder, is_partial=is_partial)
			obj = json.load(json_file, cls=decoder_cls) 
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
        Default class for module configuration.
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