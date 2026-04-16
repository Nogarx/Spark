#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import json
import numpy as np
import jax
import jax.numpy as jnp
import warnings
import typing as tp
import spark.core.utils as utils
from spark.core.registry import REGISTRY
from spark.core.config import SparkConfig
from spark.core.specs import PortSpecs, PortMap, ModuleSpecs

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class SparkJSONEncoder(json.JSONEncoder):
	"""
		Custom JSON encoder to handle common types encounter in Spark.
	"""
	__version__ = '1.0'

	def __init__(self, *args, is_partial: bool = False, **kwargs) -> None:
		self._is_partial = is_partial
		super().__init__(*args, **kwargs)

	def encode(self, obj):
		wrapped = {
			'__version__': self.__version__,
			'__data__': obj
		}
		return super().encode(wrapped)

	def default(self, obj) -> dict[str, tp.Any]:
		# Encode arrays
		if isinstance(obj, (jax.Array, np.ndarray)):
			return {
				'__type__': 'array',
				'dtype': obj.dtype.name,
				'shape': list(obj.shape),
				'data': obj.tolist()
			}
		# Encode spark configs
		if isinstance(obj, SparkConfig):
			# NOTE: Using the to_dict method will destroy all the metadata of nested classes.
			# We need to let the encoder to naturally reach config leaves
			return {
				'__type__': REGISTRY.CONFIG.get_by_cls(obj.__class__).name,
				'__cfg__': {k: v for k,v in obj}
			}
		# Encode spark specs. 
		# NOTE: Order matters!
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
			# TODO: Somewhere in the encoding/decoding dtypes are transformed to plain np.dtypes 
			# rather than np.dtypes('#').type. Below is a temporary patch
			return {
				'__type__': 'dtype',
				'name': obj.__name__ if isinstance(obj, type) else obj.type.__name__,
			}
		# Default handler
		return super().default(obj)
	
#-----------------------------------------------------------------------------------------------------------------------------------------------#

T = tp.TypeVar('T')

class SparkJSONDecoder(json.JSONDecoder):
	"""
		Custom JSON decoder to handle common types encounter in Spark.
	"""
	__supported_versions__ = {'1.0'}

	def __init__(self, *args, ignore_version: bool = False, is_partial: bool = False, **kwargs) -> None:
		self._ignore_version = ignore_version
		self._is_partial = is_partial
		super().__init__(object_hook=self.object_hook, *args, **kwargs)

	def object_hook(self, obj: dict) -> tp.Any:
		# Intercept top-level wrapper:
		if '__version__' in obj and '__data__' in obj:
			version = obj.get('__version__')
			# Sanity checks
			if not self._ignore_version:
				if version not in self.__supported_versions__:
					raise ValueError(
						f'Unsupported version: {version}. '
						f'Use the flag "ignore_version=True" if you wish to continue at your own risk.'
					)
			else:
				if version not in self.__supported_versions__:
					warnings.warn(
						f'Warning: Unsupported version {version}, decoding may fail unexpectedly.'
					)
			return obj.get('__data__')

		# Decode arrays
		if obj.get('__type__') == 'array':
			return np.array(obj.get('data'), dtype=obj.get('dtype')).reshape(obj.get('shape'))
		# Decode dtypes
		if obj.get('__type__') == 'dtype':
			return np.dtype(obj.get('name')).type
		# Decode modules cls
		if isinstance(obj, dict) and obj.get('__module_type__'):
			module_type: str | None = obj.get('__module_type__')
			subregistry: str | None = obj.get('__subregistry__')
			reg = getattr(REGISTRY, subregistry).get(module_type)
			if not reg:
				raise KeyError(f'There is no module with name "{module_type}" in the registry.')
			return reg.class_ref
		# Decode spark configs
		if obj.get('__cfg__'):
			config_type: str | None = obj.get('__type__')
			reg = REGISTRY.CONFIG.get(config_type)
			if not reg:
				raise KeyError(f'There is no registered configuration "{config_type}" in the registry.')
			config_data = obj.get('__cfg__')
			return reg.class_ref.partial(**config_data)
		# Decode spark specs
		if obj.get('__type__') == 'port_specs':
			return self._decode_spec(PortSpecs, obj)
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
		return _type.from_dict(data, is_partial=self._is_partial)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################