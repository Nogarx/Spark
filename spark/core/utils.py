#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import re
import string
import numpy as np
import typing as tp
import collections.abc
import copy 
from enum import Enum

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def normalize_str(s: str) -> str:
    """
        Converts any string into a consistent lowercase_snake_case format.

        Args:
            s: str, string to normalize

        Returns:
            str, normalized string
    """
    if not isinstance(s, str) or not s:
        raise TypeError(
            f's must be a non-empty string, got \"{s}\".'
        )
    # Insert underscores between acronyms and other words.
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    # Insert underscores between lowercase letters and uppercase letters.
    s = re.sub(r'([a-z])([A-Z])', r'\1_\2', s)
    # Replace any spaces or hyphens with a single underscore.
    s = re.sub(r'[-\s]+', '_', s)
    # Convert the whole string to lowercase.
    return s.lower()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def to_human_readable(s: str) -> str:
    """
        Converts a string from various programming cases into a human-readable format.

        Input:
            s: str, string to normalize
            
        Output:
            str, human readable string
    """
    # Sanity check
    if not isinstance(s, str) or not s:
        raise TypeError('\"s\" must be a non-empty string.')
    # Insert underscores between acronyms and other words.
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    # Insert underscores between lowercase letters and uppercase letters.
    s = re.sub(r'([a-z])([A-Z])', r'\1_\2', s)
    # Replace any spaces or hyphens with a single underscore.
    s = re.sub(r'[-\s]+', '_', s)
    # Replace all underscores with spaces.
    return s.replace('_', ' ').capitalize()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_einsum_labels(num_dims: int, offset: int = 0) -> str:
    """
        Generates labels for a generalized dot product using Einstein notation.

        Args:
            num_dims: int, number of dimensions (labels) to generate
            offset: int, initial dimension (label) offset

        Returns:
            str, a string with num_dims different labels, skipping the first offset characters 
    """
    if (offset + num_dims) > len(string.ascii_letters):
        raise ValueError(
            f'Requested up to {offset + num_dims} symbols but it is only possible to represent up to {len(string.ascii_letters)} '
            f'different symbols. If this was intentional consider defining a custom label map.'
        )
    return string.ascii_letters[offset:offset+num_dims]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_axes_einsum_labels(axes: tuple[int, ...], ignore_repeated:bool = False) -> str:
	"""
		Generates labels for a generalized dot product using Einstein notation.

		Args:
			axes: tuple[int, ...], requested dimensions (labels) to generate

		Returns:
			str, a string with num_dims different labels, skipping the first offset characters 
	"""
	
	if any([ax < 0 for ax in axes]):
		raise ValueError(
			f'\"axes\" out of bounds, expected all axis to be positive. '
		)
	
	if any([ax >= len(string.ascii_letters) for ax in axes]):
		raise ValueError(
			f'\"axes\" out of bounds, it is only possible to represent up to {len(string.ascii_letters)-1} symbols. '
			f'If this was intentional consider defining a custom label map.'
		)
	if (not ignore_repeated) and len(set(ax for ax in axes)) != len(axes):
		raise ValueError(
			f'Requested two labels for the same axis. If this was intended use the flag \"ignore_repeated=True\".'
		)
	return ''.join([string.ascii_letters[ax] for ax in axes])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def validate_shape(obj: tp.Any) -> tuple[int, ...]:
    """
        Verifies that the object is broadcastable to a valid shape (tuple of integers).
        Returns the shape.

        Args:
            obj: tp.Any: the instance to validate

        Returns:
            list[tuple[int, ...]], the shape
    """
    # Sanity checks
    if isinstance(obj, int):
        return tuple([obj])
    elif isinstance(obj, collections.abc.Iterable) and len(obj) > 0:
        for element in obj:
            if not isinstance(element, int):
                raise TypeError(
                    f'Shape elements must be integers, but found type \"{type(element).__name__}\".'
                )
            if element < 0:
                raise ValueError(
                    f'Shape elements cannot be negative, but found value \"{element}\".'
                )
    else:
        raise TypeError(f'Expected obj to be either an iterable or an int, got \"{obj}\".')
    # Cast to tuple
    return tuple(obj)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def validate_list_shape(obj: tp.Any) -> list[tuple[int, ...]]:
    """
        Verifies that the object is broadcastable to a valid list ofshape (a list of tuple of integers).
        Returns the list of shapes.

        Args:
            obj: tp.Any: the instance to validate

        Returns:
            list[tuple[int, ...]], the list of shapes
    """
    # Sanity checks
    if not isinstance(obj, collections.abc.Iterable) or len(obj) == 0:
        raise TypeError(
            f'Expected obj to be an Iterable of Iterables (e.g. list of lists), got \"{obj}\"'
        )
    if is_shape(obj):
        raise TypeError(
            f'Ambiguous input: obj can also be broadcasted to shape, got \"{obj}\". '
            f'To prevent bugs a list of shapes is only broadcastable from an Iterable of Iterables (e.g. list of lists).'
        )
    # Cast to list of shapes
    return [validate_shape(e) for e in obj]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_shape(obj: tp.Any) -> bool:
    """
        Checks if the obj is broadcastable to a shape.

        Args:
            obj: tp.Any: the instance to check.

        Returns:
            bool, True if the object is broadcastable to a shape, False otherwise.
    """
    try: 
        validate_shape(obj)
        return True
    except:
        return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_list_shape(obj: tp.Any) -> bool:
    """
        Checks if the obj is broadcastable to a shape.

        Args:
            obj: tp.Any: the instance to check.

        Returns:
            bool, True if the object is broadcastable to a list of shapes, False otherwise.
    """
    try: 
        validate_list_shape(obj)
        return True
    except:
        return False


#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_dict_of(obj: tp.Any, value_cls: type[tp.Any], key_cls: type[tp.Any] = str) -> bool:
    """
        Check if an object instance is of 'dict[key_cls, value_cls]'.

        Args:
            obj: tp.Any: the instance to check.
            key_cls: type[tp.Any], the class to compare keys against.
            value_cls: type[tp.Any], the class to compare values against.

        Returns:
            bool, True if the object is an instance of 'dict[key_cls, value_cls]', False otherwise.
    """
    if not isinstance(key_cls, type):
        raise TypeError(
            f'Expected \"key_cls\" to be of a type but got \"{key_cls}\".'
        )
    if not isinstance(value_cls, type):
        raise TypeError(
            f'Expected \"value_cls\" to be of a type but got \"{key_cls}\".'
        )
    if isinstance(obj, dict):
        if all(isinstance(k, key_cls) and isinstance(v, value_cls) for k, v in obj.items()):    
            return True
    return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_list_of(obj: tp.Any, cls: type[tp.Any]) -> bool:
    """
        Check if an object instance is of 'list[cls]'.

        Args:
            obj: tp.Any, the instance to check.
            cls: type[tp.Any], the class to compare values against.

        Returns:
            bool, True if the object is an instance of 'list[cls]', False otherwise.
    """
    if not isinstance(cls, type):
        raise TypeError(
            f'Expected \"cls\" to be of a type but got \"{cls}\".'
        )
    if isinstance(obj, list):
        if all(isinstance(x, cls) for x in obj):    
            return True
    return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_dtype(obj: tp.Any) -> bool:
	"""
		Check if an object is a 'DTypeLike'.

		Args:
			obj (tp.Any): The instance to check.
		Returns:
			bool, True if the object is a 'DTypeLike', False otherwise.
	"""
	try:
		if np.isdtype(obj, ('numeric', 'bool')):
			return True
	except: 
		pass
	return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_float(obj: tp.Any) -> bool:
	"""
		Check if an object is a 'DTypeLike'.

		Args:
			obj (tp.Any): The instance to check.
		Returns:
			bool, True if the object is a 'DTypeLike', False otherwise.
	"""
	try:
		if np.isdtype(obj, ('real floating',)):
			return True
	except: 
		pass
	return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def ascii_tree(text: str) -> str:
    """
        Build an ASCII tree from indentation-based text.
        Each level is inferred from leading spaces.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return ''

    # Map distinct indentation widths to discrete depth levels
    indents = sorted({len(l) - len(l.lstrip()) for l in lines})
    depth_map = {n: i for i, n in enumerate(indents)}

    # Build structure as list of (depth, name)
    items = [(depth_map[len(l) - len(l.lstrip())], l.strip()) for l in lines]
    stack, tree = [], []

    for depth, name in items:
        node = {'name': name, 'children': []}
        if depth == 0:
            tree.append(node)
            stack = [node]
        else:
            parent = stack[depth - 1]
            parent['children'].append(node)
            if len(stack) > depth:
                stack[depth] = node
                stack = stack[:depth + 1]
            else:
                stack.append(node)

    def render(nodes, prefix='', is_root=True):
        out = []
        for i, n in enumerate(nodes):
            last = i == len(nodes) - 1
            connector = '' if is_root else ('└── ' if last else '├── ')
            out.append(f'{prefix}{connector}{n['name']}')
            if n['children']:
                ext = '' if is_root else ('    ' if last else '│   ')
                out += render(n['children'], prefix + ext, is_root=False)
        return out

    return '\n'.join(render(tree))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InheritanceStatus(Enum):
    inheriting = 1
    receiving = 2
    can_inherit = 3
    cannot_inherit = 4

class InheritanceTree:

    def __init__(self, path: list[str] = ['_']):
        self._is_valid = False
        self._current_path = path
        self._leaves: dict[str, InheritanceStatus] = {}
        self._branches: dict[str, InheritanceTree] = {}

    def __repr__(self,) -> str:
        if not self._is_valid:
            self.validate()
        r = self._parse_tree_with_spaces(0)
        return ascii_tree(r)

    def _parse_tree_with_spaces(self, current_depth: int) -> str:
        rep = current_depth * ' ' + f'{self._current_path[-1]}\n'
        for l, s in self._leaves.items():
            rep += (current_depth + 1) * ' ' + f'{l}: {str(s)}\n'
        for _, t in self._branches.items():
            rep += t._parse_tree_with_spaces(current_depth+1)
        return rep

    def add_leaf(self, path: list[str], status: InheritanceStatus):
        # Make a copy of the path to prevent overrides
        path = copy.deepcopy(path)
        if len(path) == 1:
            # Add leave to current level.
            self._leaves[path[0]] = status
        elif len(path) > 1:
            # Consume leading path string.
            branch = path.pop(0)
            # Allow adding branches to simplify usage.
            if branch not in self._branches:
                self.add_branch([branch])
            self._branches[branch].add_leaf(path, status)
        else: 
            raise ValueError(
                f'Invalid path, got: {path}. Path must point to final leaf.'
            )
        self._is_valid = False
        
    def add_branch(self, path: list[str]):
        # Make a copy of the path to prevent overrides
        path = copy.deepcopy(path)
        if len(path) == 1:
            # Add branch to current level.
            self._branches[path[0]] = InheritanceTree(self._current_path + path)
        elif len(path) > 1:
            # Consume leading path string.
            branch = path.pop(0)
            # Allow adding branches recursively to simplify usage.
            if branch not in self._branches:
                self._branches[branch] = InheritanceTree(self._current_path + branch)
            self._branches[branch].add_branch(path)
        else: 
            raise ValueError(
                f'Invalid path, got: {path}. Path must point to final branch.'
            )
        
    def validate(self, inheriting_labels: set = set()):
        inheriting_labels = copy.deepcopy(inheriting_labels)
        if self._is_valid:
            return
        # If there are no branches, leaves cannot inherit.
        if len(self._branches) == 0:
            for l in self._leaves.keys():
                self._leaves[l] = InheritanceStatus.cannot_inherit
        # Check that status is not overriden by another label.
        else:
            for l, s in self._leaves.items():
                if l in inheriting_labels:
                    # Upper variable is forcing this one to inherit its value
                    self._leaves[l] = InheritanceStatus.receiving
                else:
                    # Check for inheritance.
                    if s == InheritanceStatus.inheriting:
                        inheriting_labels.add(l)
        # Validate branches:
        for b in self._branches.keys():
            self._branches[b].validate(inheriting_labels)
        # Flag
        self._is_valid = True

    def get_leaf_status(self, path: list[str]) -> InheritanceStatus:
        # Make a copy of the path to prevent overrides
        path = copy.deepcopy(path)
        if len(path) == 1:
            # Add branch to current level.
            return self._leaves[path[0]]
        elif len(path) > 1:
            branch = path.pop(0)
            return self._branches[branch].get_leaf_status(path)

    def is_inheriting(self, path):
        if not self._is_valid:
            self.validate()
        # Get leaf status
        status = self.get_leaf_status(path)
        if status == InheritanceStatus.inheriting:
            return True
        return False
    
    def is_receiving(self, path):
        if not self._is_valid:
            self.validate()
        # Get leaf status
        status = self.get_leaf_status(path)
        if status == InheritanceStatus.receiving:
            return True
        return False

    def can_inheriting(self, path):
        if not self._is_valid:
            self.validate()
        # Get leaf status
        status = self.get_leaf_status(path)
        if status == InheritanceStatus.cannot_inherit:
            return False
        return True
    
    def to_dict(self,) -> dict:
        if not self._is_valid:
            self.validate()
        # Collect leaves
        return {
            **{l: s.value for l, s in self._leaves.items()},
            **{b: t.to_dict() for b, t in self._branches.items()}
        }
    
    @classmethod
    def from_dict(self, d: dict, path: list[str] = ['_']):
        tree = InheritanceTree(path)
        for k, v in d.items():
            if isinstance(v, int):
                tree.add_leaf([k], InheritanceStatus(v))
            elif isinstance(v, dict):
                tree.add_branch([k])
                tree._branches[k] = InheritanceTree.from_dict(v, path=path+[k])
        return tree

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
