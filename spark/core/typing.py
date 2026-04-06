#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import inspect
import types
import typing as tp
import jax
import numpy as np
import functools
from numpy.typing import ArrayLike, DTypeLike
from jax.typing import ArrayLike as JaxArrayLike
from jax.typing import DTypeLike as JaxDTypeLike

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def is_dtype_like(obj: tp.Any) -> bool:
	"""
		Check if an object is a 'DTypeLike'.

		Args:
			obj (tp.Any): The instance to check.
		Returns:
			bool, True if the object is a 'DTypeLike', False otherwise.
	"""
	try:
		if isinstance(obj, str):
			np.dtype(obj)
			return True
		elif obj in (int, float, bool):
			return True
		elif np.isdtype(obj, ('numeric', 'bool')):
			return True
	except: 
		pass
	return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_array_like(obj: tp.Any) -> bool:
	"""
		Check if an object is a 'ArrayLike'.

		Args:
			obj (tp.Any): The instance to check.
		Returns:
			bool, True if the object is a 'ArrayLike', False otherwise.
	"""
	try:
		if isinstance(obj, (np.ndarray, jax.Array)):
			return True
	except: 
		pass
	return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# The following is a solution adapted from: 
# https://github.com/Mik-813/nested-type-checker

def is_object_of_type(obj, _type) -> bool:
	# Get root type
	origin_type = _type if tp.get_origin(_type) is None else tp.get_origin(_type)

	# NOTE: Since not all annotations are avialble at the begining we need to use its str name
	#weak_typing_mode = True if isinstance(_type, str) else False 

	# Get children types
	type_children = tp.get_args(_type)

	# Case 1: obj is None
	if origin_type is None:
		return obj is None
	#elif weak_typing_mode and origin_type == type(None).__name__:
	#	return obj is None

	# Case 2: obj is a union, check against every child
	if origin_type is tp.Union or origin_type is types.UnionType:

		# Check for ArrayLike/DTypeLike unions
		if _type == DTypeLike or _type == JaxDTypeLike:
			return is_dtype_like(obj)
		elif _type == ArrayLike or _type == JaxArrayLike:
			return is_array_like(obj)
		
		# Standard check
		else:
			return any(
				[is_object_of_type(obj, t) for t in type_children]
			)
	
	# Case 3: obj is optional, check against every child plus None
	if origin_type is tp.Optional:
		return any(
			[is_object_of_type(obj, t) for t in (*type_children, None)]
		)
	
	# Case 4: type is Any, trivial case.
	if origin_type is tp.Any:
		return True

	# Case 5: Check for numpy/jax isolated annotations. 
	# NOTE: ArrayLike / DTypeLike include extra annotation that are likely never going to be used 
	# outside those contexts which could lead to potentially false positive/negatives.
	if origin_type in (np.ndarray, jax.Array):
		return is_array_like(obj)
	if origin_type is np.dtype:
		return is_dtype_like(obj)
	
	# Check that obj matches root type.
	if not isinstance(obj, origin_type):
		return False
	# Case 6: if doesn't have childs we are done (root_type is type, per previous check).
	if not type_children:
		return True

	# Case 7: object is iterable validate children types.
	if issubclass(origin_type, tp.Iterable):

		# Subcase 1: obj is tuple
		if issubclass(origin_type, tp.Tuple):
			# Tuple is a sequence of child types
			if len(type_children) == 2 and isinstance(type_children[1], types.EllipsisType):
				# Validate each child against the first type
				for child in obj:
					if not is_object_of_type(child, type_children[0]):
						return False
			# Tuple is a sequence of fixed types
			elif len(type_children) == len(obj):
				for child, child_type in zip(obj, type_children):
					# Validate each children against its expected type
					if not is_object_of_type(child, child_type):
						return False
			else:
				raise RuntimeError(
					f'Undefined behaviour: Unable to validate "{obj}" against type check "{type_children}".'
				)
				#return False
			# Tuple is valid
			return True
		
		# Subcase 2: obj is a set-like object
		if any(issubclass(origin_type, t) for t in {tp.Sequence, tp.Set, tp.FrozenSet}):
			# Validate each child against the first type
			for child in obj:
				if not is_object_of_type(child, type_children[0]):
					return False
			# set-like object is valid
			return True

		# Subcase 3: obj is a dictionary
		if issubclass(origin_type, tp.Mapping):
			# Check if dictionary is annotated.
			obj_item_columns = obj.keys(), obj.values() if obj.keys() else ()
			if len(obj_item_columns) != len(type_children):
				return False
			# If annotated check each entry against the expected types
			for obj_item_column, type_child in zip(obj_item_columns, type_children):
				for obj_item in obj_item_column:
					if not is_object_of_type(obj_item, type_child):
						return False
			# dict is valid
			return True

	# Case 8: origin_type is a raw type
	if origin_type == type:
		return any(
			[issubclass(obj, t) for t in type_children]
		)

	# Case 9: if object is a function.
	# NOTE: Actual function validation is practicaly impossible to pull out without without formal verification
	# This case only inspects that the registered signature matches with the provided signature (_type)
	if issubclass(origin_type, tp.Callable):
		signature = inspect.signature(obj)
		return_type = signature.return_annotation
		param_types = tuple(param.annotation for param in signature.parameters.values())
		# Check output type against expected type
		specified_return_type = (
			None if type_children[1] is types.NoneType else type_children[1]
		)
		if (
			specified_return_type is not tp.Any
			and return_type != specified_return_type
		):
			return False
		# Check each input against expected types
		if len(param_types) != len(type_children[0]):
			return False
		for i, specified_param_type in enumerate(type_children[0]):
			if specified_param_type is tp.Any:
				continue
			specified_param_type = (
				None if specified_param_type is types.NoneType else specified_param_type
			)
			if param_types[i] != specified_param_type:
				return False
		# Callable is valid
		return True

	# Default case: we exhausted every check but we were unable to validate the obj.
	raise RuntimeError(
		f'Undefined behaviour: Unable to validate "{obj}" against type check "{_type}".'
	)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def enforce_annotations(fn, strict: bool = True, validate_keys: bool = True) -> tp.Callable:
	# Get function signature
	parameters = inspect.signature(fn).parameters
	parameters_names = tuple(param for param in parameters.keys())

	# Common error method for args/kwargs
	def _get_param_error(param: inspect.Parameter, value: tp.Any) -> str:
		# Check for ArrayLike/DTypeLike
		if param.annotation == DTypeLike or param.annotation == JaxDTypeLike:
			expected_annotation = 'DTypeLike'
		elif param.annotation == ArrayLike or param.annotation == JaxArrayLike:
			expected_annotation = 'ArrayLike'
		else:
			expected_annotation = param.annotation
		return f'Positional argument "{param.name}" is of type {type(value)} but expected {expected_annotation}.'

	# Wrapper definition
	@functools.wraps(fn)
	def wrapper(*args, **kwargs):
		errors = list()
		# Validate positional arguments
		for i in range(len(args)):
			param = parameters[parameters_names[i]]
			value = args[i]

			# Skip unimportant parameters.
			if param.name in ['self', 'cls']: 
				continue

			# Skip non-annotated arguments
			if param.annotation is inspect._empty:
				if strict:
					errors.append(
						f'Positional argument "{param.name}" does not define an expected type. Use "strict=False" if this was intended.'
					)
				continue
			# Validate object
			if not is_object_of_type(value, param.annotation):
				errors.append(_get_param_error(param, value))

		# Validate keyworded argument keys
		if validate_keys:
			for key in kwargs.keys():
				if not key in parameters_names:
					errors.append(
						f'Keyword argument "{key}" is not part of the function signature. Use "validate_keys=False" if this was intended.'
					)

		# Validate keyword values
		missing_keys = parameters_names[len(args):]
		for key in missing_keys:
			param = parameters[key]
			# Skip unimportant parameters.
			if param.name in ['self', 'cls']: 
				continue
			# Try get argument from kwargs
			try:
				value = kwargs[key]
			except:
				value = None if parameters[key].default is types.NoneType else parameters[key].default
			# Skip non-annotated arguments
			if param.annotation is inspect._empty:
				if strict:
					errors.append(
						f'Keyword argument "{param.name}" does not define an expected type. Use "strict=False" if this was intended.'
					)
				continue
			# Validate object
			if not is_object_of_type(value, param.annotation):
				errors.append(_get_param_error(param, value))
				
		# Raise errors.
		if len(errors):
			raise TypeError('\n'.join(errors))

		# Call original function.
		return fn(*args, **kwargs)

	return wrapper

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################