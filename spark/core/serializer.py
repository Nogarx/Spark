#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import json
import numpy as np
import jax
import jax.numpy as jnp
import warnings
import typing as tp
import spark.core.utils as utils
from jax.typing import DTypeLike
from spark.core.registry import REGISTRY
from spark.core.config import BaseSparkConfig, InitializableField
from spark.core.specs import PortSpecs, InputSpec, OutputSpec, PortMap, ModuleSpecs

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

T = tp.TypeVar('T')

class SparkJSONEncoder(json.JSONEncoder):
	"""
		Custom JSON encoder to handle common types encounter in Spark.
	"""
	__version__ = '1.0'

	def encode(self, obj):
		wrapped = {
			'__version__': self.__version__,
			'__data__': obj
		}
		return super().encode(wrapped)

	#def iterencode(self, o, _one_shot=False):
	#	return super().iterencode(self._preprocess(o), _one_shot)

	def default(self, obj):
		# Encode jax arrays
		if isinstance(obj, (jax.Array, jnp.ndarray)):
			return {
				'__type__': 'jax_array',
				'dtype': obj.dtype.name,
				'shape': list(obj.shape),
				'data': obj.tolist()
			}
		# Encode numpy arrays
		if isinstance(obj, np.ndarray):
			return {
				'__type__': 'numpy_array',
				'dtype': obj.dtype.name,
				'shape': list(obj.shape),
				'data': obj.tolist()
			}
		# Encode spark configs
		if isinstance(obj, BaseSparkConfig):
			return {
				'__type__': REGISTRY.CONFIG.get_by_cls(obj.__class__).name,
				'__cfg__': obj.to_dict(),
			}
		# Encode spark specs. 
		# NOTE: Order matters!
		if isinstance(obj, InputSpec):
			return  {
				'__type__': 'input_specs',
				'__data__': obj.to_dict(),
			}
		if isinstance(obj, OutputSpec):
			return  {
				'__type__': 'output_specs',
				'__data__': obj.to_dict(),
			}
		if isinstance(obj, PortSpecs):
			return  {
				'__type__': 'port_specs',
				'__data__': obj.to_dict(),
			}
		if isinstance(obj, PortMap):
			return  {
				'__type__': 'port_map',
				'__data__': obj.to_dict(),
			}
		if isinstance(obj, ModuleSpecs):
			return  {
				'__type__': 'module_specs',
				'__data__': obj.to_dict(),
			}
		# Encode jax/numpy dtypes
		if utils.is_dtype(obj):
			return {
				'__type__': 'dtype',
				'name': obj.__name__,
			}
		# Default handler
		return super().default(obj)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkJSONDecoder(json.JSONDecoder):
	"""
		Custom JSON decoder to handle common types encounter in Spark.
	"""
	__supported_versions__ = {'1.0'}

	def __init__(self, *args, ignore_version: bool = False, **kwargs):
		self._ignore_version = ignore_version
		super().__init__(object_hook=self.object_hook, *args, **kwargs)

	def object_hook(self, obj: dict):
		# Intercept top-level wrapper:
		if '__version__' in obj and '__data__' in obj:
			version = obj.get('__version__')
			# Sanity checks
			if not self._ignore_version:
				if version not in self.__supported_versions__:
					raise ValueError(
						f'Unsupported version: {version}. '
						f'Use the flag \"ignore_version=True\" if you wish to continue at your own risk.'
					)
			else:
				if version not in self.__supported_versions__:
					warnings.warn(
						f'Warning: Unsupported version {version}, decoding may fail unexpectedly.'
					)
			return obj.get('__data__')

		# Decode jax arrays
		if obj.get('__type__') == 'jax_array':
			return jnp.array(obj.get('data'), dtype=obj.get('dtype')).reshape(obj.get('shape'))
		# Decode numpy arrays
		if obj.get('__type__') == 'numpy_array':
			return np.array(obj.get('data'), dtype=obj.get('dtype')).reshape(obj.get('shape'))
		# Decode numpy/jax dtypes
		if obj.get('__type__') == 'dtype':
			return np.dtype(obj.get('name')).type
		# Decode payload and module types
		if isinstance(obj, dict) and obj.get('__payload_type__'):
			payload_type: str | None = obj.get('__payload_type__')
			if not payload_type or not isinstance(payload_type, str):
				raise TypeError(f'Expected \"__payload_type__\" to be of type \"str\", but got {payload_type}')
			reg = REGISTRY.PAYLOADS.get(payload_type)
			if not reg:
				raise KeyError(f'There is no payload with name \"{payload_type}\" in the registry.')
			return reg.class_ref
		if isinstance(obj, dict) and obj.get('__module_type__'):
			module_type: str | None = obj.get('__module_type__')
			if not module_type or not isinstance(module_type, str):
				raise TypeError(f'Expected \"__module_type__\" to be of type \"str\", but got {module_type}')
			reg = REGISTRY.MODULES.get(module_type)
			if not reg:
				raise KeyError(f'There is no module with name \"{module_type}\" in the registry.')
			return reg.class_ref
		# Decode spark configs
		if obj.get('__cfg__'):
			config_type: str | None = obj.get('__type__')
			if not config_type or not isinstance(config_type, str):
				raise TypeError(f'Expected \"__type__\" to be of type \"str\", but got {config_type}')
			reg = REGISTRY.CONFIG.get(config_type)
			if not reg:
				raise KeyError(f'There is no config with name \"{config_type}\" in the registry.')
			config_data = obj.get('__cfg__')
			if not isinstance(config_data, dict):
				raise TypeError(f'Expected \"__cfg__\" to be of type \"dict\", but got {config_data}')
			return reg.class_ref(**config_data)
			#return cls.from_dict(obj.get('__cfg__'))
		# Decode spark specs
		if obj.get('__type__') == 'port_specs':
			return self._decode_spec(PortSpecs, obj)
		if obj.get('__type__') == 'input_specs':
			return self._decode_spec(InputSpec, obj)
		if obj.get('__type__') == 'output_specs':
			return self._decode_spec(OutputSpec, obj)
		if obj.get('__type__') == 'port_map':
			return self._decode_spec(PortMap, obj)
		if obj.get('__type__') == 'module_specs':
			return self._decode_spec(ModuleSpecs, obj)
		# Default handler
		return obj
	
	def _decode_spec(self, _type: type[T], obj: dict) -> T:
		data = obj.get('__data__')
		if not isinstance(data, dict):
			raise TypeError(f'Expected \"__data__\" to be of type \"dict\", but got {data}')
		return _type.from_dict(data)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################