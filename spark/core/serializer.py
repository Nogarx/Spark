#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import json
import numpy as np
import jax.numpy as jnp
import warnings
from jax.typing import DTypeLike
from spark.core.registry import REGISTRY
from spark.core.config import BaseSparkConfig
from spark.core.shape import Shape, ShapeCollection
from spark.core.specs import PortSpecs, InputSpec, OutputSpec, PortMap, ModuleSpecs

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

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

	def iterencode(self, o, _one_shot=False):
		return super().iterencode(self._preprocess(o), _one_shot)

	def _preprocess(self, obj):
		# Encode shape
		if isinstance(obj, Shape):
			return {
				'__type__': 'shape',
				'data': list(obj),
			}
		# Encode shape collections
		if isinstance(obj, ShapeCollection):
			return {
				'__type__': 'shape_collection',
				'data': list(list(o) for o in obj),
			}
		elif isinstance(obj, dict):
			return {k: self._preprocess(v) for k, v in obj.items()}
		elif isinstance(obj, (list, tuple)):
			return [self._preprocess(v) for v in obj]
		else:
			return obj

	def default(self, obj):
		# Encode shape
		if isinstance(obj, Shape):
			return {
				'__type__': 'shape',
				'data': list(obj),
			}
		# Encode shape collections
		if isinstance(obj, ShapeCollection):
			return {
				'__type__': 'shape_collection',
				'data': list(list(o) for o in obj),
			}
		# Encode jax arrays
		if isinstance(obj, jnp.ndarray):
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
				'__cfg__': obj._to_dict(),
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
		# NOTE: isdtype thinks BaseSparkConfig, PortSpecs, InputSpec, OutputSpec, PortMap and ModuleSpecs are dtypes!. 
		# isdtype should be the last checks.
		# Encode jax dtypes
		if jnp.isdtype(obj, ('numeric', 'bool')):
			return {
				'__type__': 'jax_dtype',
				'name': obj.__name__,
			}
		# Encode numpy dtypes
		if np.isdtype(obj, ('numeric', 'bool')):
			return {
				'__type__': 'numpy_dtype',
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
						f'⚠️ Warning: Unsupported version {version}, decoding may fail unexpectedly.'
					)
			return obj.get('__data__')

		# Decode jax arrays
		if obj.get('__type__') == 'jax_array':
			return jnp.array(obj.get('data'), dtype=obj.get('dtype')).reshape(obj.get('shape'))
		# Decode numpy arrays
		if obj.get('__type__') == 'numpy_array':
			return np.array(obj.get('data'), dtype=obj.get('dtype')).reshape(obj.get('shape'))
		# Decode jax dtypes
		if obj.get('__type__') == 'jax_dtype':
			return jnp.dtype(obj.get('name')).type
		# Decode jax dtypes
		if obj.get('__type__') == 'numpy_dtype':
			return np.dtype(obj.get('name')).type
		# Decode shapes
		if obj.get('__type__') == 'shape':
			return Shape(obj.get('data'))
		# Decode shape collections
		if obj.get('__type__') == 'shape_collection':
			return ShapeCollection(obj.get('data'))
		# Decode payload and module types
		if isinstance(obj, dict) and obj.get('__payload_type__'):
			return REGISTRY.PAYLOADS.get(obj.get('__payload_type__')).class_ref
		if isinstance(obj, dict) and obj.get('__module_type__'):
			return REGISTRY.MODULES.get(obj.get('__module_type__')).class_ref
		# Decode spark configs
		if obj.get('__cfg__'):
			cls: BaseSparkConfig = REGISTRY.CONFIG.get(obj.get('__type__')).class_ref
			return cls(**obj.get('__cfg__'))
			#return cls.from_dict(obj.get('__cfg__'))
		# Decode spark specs
		if obj.get('__type__') == 'port_specs':
			return PortSpecs.from_dict(obj.get('__data__'))
		if obj.get('__type__') == 'input_specs':
			return InputSpec.from_dict(obj.get('__data__'))
		if obj.get('__type__') == 'output_specs':
			return OutputSpec.from_dict(obj.get('__data__'))
		if obj.get('__type__') == 'port_map':
			return PortMap.from_dict(obj.get('__data__'))
		if obj.get('__type__') == 'module_specs':
			return ModuleSpecs.from_dict(obj.get('__data__'))
		# Default handler
		return obj

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Serializer:

    def __init__(self):
        pass

    def serialize():
        pass

    def _serialize_shape(shape: Shape) -> list[int]:
        return list(shape)
    
    def _deserialize_shape(shape: list[int]) -> Shape:
        return tuple(shape)

    def _serialize_dtype(dtype: DTypeLike) -> str:
        return dtype.name
    
    def _deserialize_dtype(shape: str) -> DTypeLike:
        return jnp.dtype(shape)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################