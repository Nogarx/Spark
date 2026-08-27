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
from spark.core.config import SparkConfig, StaticValue
from spark.core.specs import PortSpecs, PortMap, ModuleSpecs

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

METADATA_KEY = '__metadata__'
"""
    Where a file keeps what was written beside the configuration. It sits next to it rather than in it, so
    that decoding answers the configuration alone and a reader that knows nothing of it reads the file.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class SparkJSONEncoder(json.JSONEncoder):
	"""
		JSON encoder for the types a Spark configuration holds.

		Handles configurations, module specifications, port maps, dtypes, jax arrays, enums and
		registered classes, which are written by their registered name rather than by value.

		See Also
		--------
		SparkJSONDecoder : Reads back what this writes.
	"""
	__version__ = '1.0'

	def __init__(self, *args, metadata: dict[str, tp.Any] | None = None, **kwargs) -> None:
		self._metadata = metadata
		super().__init__(*args, **kwargs)

	def iterencode(self, obj, _one_shot: bool = False):
		wrapped = {
			'__version__': self.__version__,
			'__data__': obj
		}
		if self._metadata is not None:
			wrapped[METADATA_KEY] = self._metadata
		return super().iterencode(wrapped, _one_shot)

	def default(self, obj) -> dict[str, tp.Any]:
		# Unwrap configuration values
		if isinstance(obj, StaticValue):
			value = obj.value
			return self.default(value) if isinstance(value, (jax.Array, np.ndarray)) else value
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
			return {
				'__type__': REGISTRY.Configs.get_by_cls(obj.__class__).name,
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
		JSON decoder for the types a Spark configuration holds.

		Reads back what `SparkJSONEncoder` writes. A class is looked up in the registry by the
		name it was written under, so the model it names has to be registered first.
	"""
	__supported_versions__ = {'1.0'}

	def __init__(self, *args, ignore_version: bool = False, **kwargs) -> None:
		self._ignore_version = ignore_version
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
		if obj.get('__type__') in ['array', 'jax_array']:
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
			return reg.get_cls()
		# Decode spark configs
		if obj.get('__cfg__'):
			config_type: str | None = obj.get('__type__')
			reg = REGISTRY.Configs.get(config_type)
			if not reg:
				raise KeyError(f'There is no registered configuration "{config_type}" in the registry.')
			config_data = obj.get('__cfg__')
			return reg.get_cls().partial(**config_data)
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
		return _type.from_dict(data)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################