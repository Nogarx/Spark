#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import re
import enum
import string
import numpy as np
import typing as tp
import collections.abc
import copy 
import dataclasses as dc
from math import prod 

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

def to_human_readable(s: str, capitalize_all: bool = False) -> str:
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
    if capitalize_all:
        return ' '.join([w.capitalize() for w in s.replace('_', ' ').split(' ')])
    else:
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

def get_einsum_dot_string(x: tuple[int, ...], y: tuple[int, ...], ignore_one_dims: bool = True, side: str = 'right') -> str:
    """
        Generates labels for a generalized dot product using Einstein notation.
            right:	(c,d)•(a,b,c,d)=(a,b) - cd,abcd->ab     |    (a,b,c,d)•(c,d)=(a,b) - abcd,cd->ab
            left:	(a,b)•(a,b,c,d)=(c,d) - ab,abcd->cd	    |	 (a,b,c,d)•(c,d)=(c,d) - abcd,ab->cd

        Args:
            x: tuple[int, ...], shape for the first variable of the dot product
            y: tuple[int, ...], shape for the second variable of the dot product
            ignore_one_dims: bool, ignore one dimensions when computing the labels (squeeze shapes), default: True
            side: str, side of the dot product, default: "right"

        Returns:
            str, a string representing the dot product operation
    """
    # Check shape is valid
    if 0 in x or 0 in y:
        raise TypeError(
            f'Invalid dot product operation dot(x,y) with dimension zero, {"x" if 0 in x else "y"}: {x if 0 in x else y}.'
        )
    # Ignore ones
    if ignore_one_dims:
        x = tuple(idx for idx in x if idx != 1)
        y = tuple(idx for idx in y if idx != 1)
    # Get labels
    side = side.lower()
    is_x_bigger = len(x) >= len(y)
    if side == 'right' or side == 'r':
        x_indices = get_einsum_labels(len(x), offset=0 if is_x_bigger else len(y)-len(x))
        y_indices = get_einsum_labels(len(y), offset=len(x)-len(y) if is_x_bigger else 0)
        z_size = len(x)-len(y) if is_x_bigger else len(y)-len(x)
        z_indices = get_einsum_labels(z_size, offset=0)
        # Validate labels / shapes
        if len(z_indices) > max(len(x),len(y)) or (is_x_bigger and y != x[-len(y):]) or (not is_x_bigger and x != y[-len(x):]):
            raise TypeError(
                f'Invalid right dot product operation dot(x,y) with shapes x:{x} and y:{y}.'
            )
    elif side == 'left' or side == 'l':
        x_indices = get_einsum_labels(len(x), offset=0)
        y_indices = get_einsum_labels(len(y), offset=0)
        z_size = len(x)-len(y) if is_x_bigger else len(y)-len(x)
        z_indices = get_einsum_labels(z_size, offset=len(y) if is_x_bigger else len(x))
        # Validate labels / shapes
        if len(z_indices) > max(len(x),len(y)) or (is_x_bigger and y != x[:len(y)]) or (not is_x_bigger and x != y[:len(x)]):
            raise TypeError(
                f'Invalid left dot product operation dot(x,y) with shapes x:{x} and y:{y}.'
            )
    else:
        raise ValueError(
            f'Invalid side value: {side}'
        )
    return f'{x_indices},{y_indices}->{z_indices}'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_einsum_dot_red_string(x: tuple[int, ...], y: tuple[int, ...], ignore_one_dims: bool = True, side: str = 'right') -> str:
    """
        Generates labels for a generalized dot reduction product using Einstein notation.
            right:	(a,b)•(a,b,c,d)=(a,b) - ab,abcd->ab     |    (a,b,c,d)•(a,b)=(a,b) - abcd,ab->ab
            left:	(c,d)•(a,b,c,d)=(c,d) - cd,abcd->cd	    |	 (a,b,c,d)•(c,d)=(c,d) - abcd,ab->ab

        Args:
            x: tuple[int, ...], shape for the first variable of the dot product
            y: tuple[int, ...], shape for the second variable of the dot product
            ignore_one_dims: bool, ignore one dimensions when computing the labels (squeeze shapes), default: True
            side: str, side of the reduction-dot product, default: "right" 

        Returns:
            str, a string representing the dot product operation
    """
    # Check shape is valid
    if 0 in x or 0 in y:
        raise TypeError(
            f'Invalid reduction-dot product operation dot(x,y) with dimension zero, {"x" if 0 in x else "y"}: {x if 0 in x else y}.'
        )
    # Ignore ones
    if ignore_one_dims:
        x = tuple(idx for idx in x if idx != 1)
        y = tuple(idx for idx in y if idx != 1)
    # Get labels
    side = side.lower()
    is_x_bigger = len(x) >= len(y)
    if side == 'right' or side == 'r':
        x_indices = get_einsum_labels(len(x), offset=0 if is_x_bigger else len(y)-len(x))
        y_indices = get_einsum_labels(len(y), offset=len(x)-len(y) if is_x_bigger else 0)
        z_size = len(y) if is_x_bigger else len(x)
        z_indices = get_einsum_labels(z_size, offset=len(x)-len(y) if is_x_bigger else len(y)-len(x))
        # Validate labels / shapes
        if len(z_indices) > max(len(x),len(y)) or (is_x_bigger and y != x[-len(y):]) or (not is_x_bigger and x != y[-len(x):]):
            raise TypeError(
                f'Invalid right reduction-dot product operation dot(x,y) with shapes x:{x} and y:{y}.'
            )
    elif side == 'left' or side == 'l':
        x_indices = get_einsum_labels(len(x), offset=0)
        y_indices = get_einsum_labels(len(y), offset=0)
        z_size = len(y) if is_x_bigger else len(x)
        z_indices = get_einsum_labels(z_size, offset=0)
        # Validate labels / shapes
        if  len(z_indices) > max(len(x),len(y)) or (is_x_bigger and y != x[:len(y)]) or (not is_x_bigger and x != y[:len(x)]):
            raise TypeError(
                f'Invalid left reduction-dot product operation dot(x,y) with shapes x:{x} and y:{y}.'
            )
    else:
        raise ValueError(
            f'Invalid side value: {side}'
        )
    return f'{x_indices},{y_indices}->{z_indices}'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_einsum_dot_exp_string(x: tuple[int, ...], y: tuple[int, ...], ignore_one_dims: bool = False, side: str = 'right') -> str:
    """
        Generates labels for a generalized dot expansion product using Einstein notation.
            right:	(a,b)•(a,b,c,d)=(a,b,c,d) - ab,abcd->abcd   |   (a,b,c,d)•(a,b)=(a,b,c,d) - abcd,ab->abcd
            left:	(c,d)•(a,b,c,d)=(a,b,c,d) - cd,abcd->abcd	|	(c,d)•(a,b,c,d)=(a,b,c,d) - abcd,cd->abcd
            none: 	(a,b)•(c,d)=(a,b,c,d) - ab,cd->abcd		    | 	(a)•(b,c,d)=(a,b,c,d) - a,bcde->abcde

        Args:
            x: tuple[int, ...], shape for the first variable of the dot product
            y: tuple[int, ...], shape for the second variable of the dot product
            ignore_one_dims: bool, ignore one dimensions when computing the labels (squeeze shapes), default: True
            side: str, side of the expansion-dot, default: "right"
                
        Returns:
            str, a string representing the dot product operation
    """
    # Check shape is valid
    if 0 in x or 0 in y:
        raise TypeError(
            f'Invalid dot-expansion product operation dot(x,y) with dimension zero, {"x" if 0 in x else "y"}: {x if 0 in x else y}.'
        )
    # Ignore ones
    if ignore_one_dims:
        x = tuple(idx for idx in x if idx != 1)
        y = tuple(idx for idx in y if idx != 1)
    # Get labels
    side = side.lower()
    is_x_bigger = len(x) >= len(y)
    if side == 'right' or side == 'r':
        x_indices = get_einsum_labels(len(x), offset=0 if is_x_bigger else len(y)-len(x))
        y_indices = get_einsum_labels(len(y), offset=len(x)-len(y) if is_x_bigger else 0)
        z_size = len(x) if is_x_bigger else len(y)
        z_indices = get_einsum_labels(z_size, offset=0)
        # Validate labels / shapes
        if (is_x_bigger and y != x[-len(y):]) or (not is_x_bigger and x != y[-len(x):]):
            raise ValueError(
                f'Invalid right dot-expansion product operation dot(x,y) with shapes x:{x} and y:{y}.'
            )
    elif side == 'left' or side == 'l':
        x_indices = get_einsum_labels(len(x), offset=0)
        y_indices = get_einsum_labels(len(y), offset=0)
        z_size = len(x) if is_x_bigger else len(y)
        z_indices = get_einsum_labels(z_size, offset=0)
        # Validate labels / shapes
        if (is_x_bigger and y != x[:len(y)]) or (not is_x_bigger and x != y[:len(x)]):
            raise ValueError(
                f'Invalid left dot-expansion product operation dot(x,y) with shapes x:{x} and y:{y}.'
            )
    elif side == 'none' or side == 'n':
        x_indices = get_einsum_labels(len(x), offset=0)
        y_indices = get_einsum_labels(len(y), offset=len(x))
        z_size = len(x) + len(y)
        z_indices = get_einsum_labels(z_size, offset=0)
    else:
        raise ValueError(
            f'Invalid side value: {side}'
        )
    return f'{x_indices},{y_indices}->{z_indices}'

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

# TODO: Extend shape promotion to allow some sensible shape blends like stacks.
def merge_shape_list(shape_list: list[tuple[int, ...]]) -> tuple[int, ...]:
    """
        Merges a list of shapes into a single shape.

        Args:
            shape_list: list[tuple[int, ...]]: the list of shapes

        Returns:
            tuple[int, ...], the merged shape
    """
    shape_list = validate_list_shape(shape_list)
    return tuple([sum([prod(s) for s in shape_list])])

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

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InheritanceFlags(enum.IntFlag):
    CAN_INHERIT = 0b1000
    IS_INHERITING = 0b0100
    CAN_RECEIVE = 0b0010
    IS_RECEIVING = 0b0001

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass
class InheritanceLeaf:
    """
        Leaf object for the InheritanceTree data structure.
    """

    name: str
    type_string: str
    inheritance_childs: list[list[str]]
    flags: InheritanceFlags = 0b0000
    break_inheritance: bool = False
    parent: InheritanceTree = None

    def __post_init__(self,) -> None:
        if isinstance(self.type_string, tp.Iterable):
            type_string = set(self.type_string)
            self.type_string = [t.__name__ if isinstance(t, type) else str(t) for t in type_string if t is not None]
        elif isinstance(self.type_string, type):
            self.type_string = self.type_string.__name__

    def __repr__(self,) -> str:
        rep = f'{self.name}\n'
        rep += ' ' + f'type_string: {self.type_string}\n'
        rep += ' ' + f'flags: {self.flags}\n'
        rep += ' ' + f'break_inheritance: {self.break_inheritance}\n'
        rep += ' ' + f'inheritance_childs:\n'
        for c in self.inheritance_childs:
            rep += 2*' ' + f'{c}\n' 
        return ascii_tree(rep)


    def to_dict(self,) -> dict:
        return {
            'name': self.name,
            'type_string': self.type_string,
            'inheritance_childs': self.inheritance_childs,
            'break_inheritance': self.break_inheritance,
            'flags': self.flags.value
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'InheritanceLeaf':
        d = copy.deepcopy(d)
        d['flags'] = InheritanceFlags(d['flags'])
        return cls(**d)

    def can_inherit(self,) -> bool:
        """
            Cheks the leaf node can inherit.
        """
        return bool(self.flags & InheritanceFlags.CAN_INHERIT)
    
    def is_inheriting(self,) -> bool:
        """
            Cheks the leaf node is inheriting.
        """
        return bool(self.flags & InheritanceFlags.IS_INHERITING)

    def can_receive(self,) -> bool:
        """
            Cheks the leaf node can receive.
        """
        return bool(self.flags & InheritanceFlags.CAN_RECEIVE)
    
    def is_receiving(self,) -> bool:
        """
            Cheks the leaf node is receiving.
        """
        return bool(self.flags & InheritanceFlags.IS_RECEIVING)

    @property
    def path(self,) -> list[str]:
        """
            Returns the path of the leaf node.
        """
        return self.parent.path + [self.name]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InheritanceTree:
    """
        Tree-like data structure to manage the inheritance status of variables in the Spark Graph Editor. 

        This data structure is used to link variables with the same names and types for simultaneous updates within the GUI. 
    """

    def __init__(self, path: list[str] = []) -> None:
        self._is_valid = False
        self._current_path = path
        self._leaves: dict[str, InheritanceLeaf] = {}
        self._branches: dict[str, InheritanceTree] = {}

    def __repr__(self,) -> str:
        if not self._is_valid:
            self.validate()
        r = self._parse_tree_with_spaces(0)
        return ascii_tree(r)

    def _parse_tree_with_spaces(self, current_depth: int) -> str:
        """
            Parses the tree with to produce a string with the appropiate format for the ascii_tree method.
        """

        rep = current_depth * ' ' + f'{self._current_path[-1]}\n' if len(self._current_path) > 0 else ''
        for l, s in self._leaves.items():
            rep += (current_depth + 1) * ' ' + f'{l}: {s.flags}\n'
        for _, t in self._branches.items():
            rep += t._parse_tree_with_spaces(current_depth+1 if len(self._current_path) > 0 else 0)
        return rep

    def add_leaf(
            self, 
            path: list[str], 
            type_string: str = '', 
            inheritance_childs: list[list[str]]=[], 
            flags: InheritanceFlags = 0b0000,
            break_inheritance: bool = False,
            **kwargs,
        ) -> None:
        """
            Adds a new leaf to the tree.

            Input:
                path: list[str], path to the new leaf node, with the last entry the name of the leaf
                type_string: str, string representation of the types this variable manages
                inheritance_childs: list[list[str]]=[], list of children that can inherit from this variable (Note: do not set by hand)
                flags: InheritanceFlags, 4-bit flags that represent inheritance possibilities (Note: do not set by hand)
                break_inheritance: bool, boolean flag to disconnect this variable from the inheritance dynamics
        """
        # Make a copy of the path to prevent overrides
        path = copy.deepcopy(path if isinstance(path, list) else list(path))
        if len(path) == 1:
            # Add leave to current level.
            self._leaves[path[0]] = InheritanceLeaf(
                name=path[0], 
                type_string=type_string,
                inheritance_childs=inheritance_childs,
                flags=flags, 
                break_inheritance=break_inheritance,
                parent=self,
            )
        elif len(path) > 1:
            # Consume leading path string.
            branch = path.pop(0)
            # Allow adding branches to simplify usage.
            if branch not in self._branches:
                self.add_branch([branch])
            self._branches[branch].add_leaf(
                path, 
                type_string=type_string, 
                flags=flags, 
                break_inheritance=break_inheritance
            )
        else: 
            raise ValueError(
                f'Invalid path, got: {path}. Path must point to final leaf.'
            )
        self._is_valid = False
        
    def add_branch(self, path: list[str]) -> None:
        """
            Adds a new branch to the tree.

            Input:
                path: list[str], path to the new branch, with the last entry the name of the branch
        """
        # Make a copy of the path to prevent overrides
        path = copy.deepcopy(path if isinstance(path, list) else list(path))
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
        
    def validate(self, inheriting_labels: dict = {}) -> None:
        """
            Validates the flags and the inheritance childs of the tree.
        """
        inheriting_labels = copy.deepcopy(inheriting_labels)
        if self._is_valid:
            return
        if len(self._branches) == 0:
            # If there are no branches, leaves cannot inherit.
            can_inherit = InheritanceFlags(0)
            is_inheriting = InheritanceFlags(0)
            for l in self._leaves.keys():
                # If label is not in inheriting_labels, leaf cannot receive.
                if l not in inheriting_labels:
                    can_receive = InheritanceFlags(0)
                    is_receiving = InheritanceFlags(0)
                else:
                    can_receive = InheritanceFlags.CAN_RECEIVE
                    is_receiving = InheritanceFlags.IS_RECEIVING if inheriting_labels[l] else InheritanceFlags(0)
                # Set leaf attributes
                if self._leaves[l].break_inheritance:
                    self._leaves[l].flags = InheritanceFlags(0)
                else:
                    self._leaves[l].flags = can_inherit | is_inheriting | can_receive | is_receiving
                self._leaves[l].inheritance_childs = []
        else:
            for l, il in self._leaves.items():
                # Get leaf inheritance_childs
                inheritance_childs = self._compute_leaf_childs(l)
                # Check if can inherit its value
                can_inherit = InheritanceFlags.CAN_INHERIT if len(inheritance_childs) > 0 else InheritanceFlags(0)
                # Preseve is_inheriting flag unless it is set on by error.
                is_inheriting = il.flags & InheritanceFlags.IS_INHERITING if can_inherit else InheritanceFlags(0)
                # If label is not in inheriting_labels, leaf cannot receive.
                if l not in inheriting_labels:
                    can_receive = InheritanceFlags(0)
                    is_receiving = InheritanceFlags(0)
                else:
                    can_receive = InheritanceFlags.CAN_RECEIVE
                    is_receiving = InheritanceFlags.IS_RECEIVING if inheriting_labels[l] else InheritanceFlags(0)
                # Set leaf attributes
                if self._leaves[l].break_inheritance:
                    self._leaves[l].flags = InheritanceFlags(0)
                else:
                    self._leaves[l].flags = can_inherit | is_inheriting | can_receive | is_receiving
                self._leaves[l].inheritance_childs = inheritance_childs

                # Add leaf to inheriting labels
                if len(inheritance_childs) > 0:
                    inheriting_labels[l] = inheriting_labels.get(l, None) or bool(is_inheriting)

        # Validate branches:
        for b in self._branches.keys():
            self._branches[b].validate(inheriting_labels)
        # Flag
        self._is_valid = True

    # NOTE: This method should be computed from deeper branches to shallow for efficiency. However, in practice Inheritance
    # trees will not have more than a few levels and a couple dozens of parameters which makes forward search acceptable.
    def _compute_leaf_childs(self, name: str, path: list[str] = []) -> list[list[str]]:
        """
            Collects the inheritance childs of a tree, relative to the current leaf.

            Input:
                name: str, leaf node name to search

            Returns:
                list[list[str]], list of inheritance childs of the leaf node
        """
        inheritance_childs = []
        # Search in subtrees only.
        if len(path) > 0:
            for l, lo in self._leaves.items():
                if l == name:
                    # Check if leaf has the break_inheritance flag
                    if not lo.break_inheritance:
                        inheritance_childs.append(path + [name])
                    break
        for b in self._branches.keys():
            inheritance_childs += self._branches[b]._compute_leaf_childs(name, [b])
        return inheritance_childs

    def get_leaf(self, path: list[str]) -> InheritanceLeaf:
        """
            Returns the status of the leaf node.

            Input:
                path: list[str], path to the leaf node, with the last entry the name of the leaf

            Returns:
                InheritanceLeaf, returns the leaf node instance.
        """
        if not self._is_valid:
            self.validate()
        # Make a copy of the path to prevent overrides
        path = copy.deepcopy(path if isinstance(path, list) else list(path))
        if len(path) == 1:
            # Add branch to current level.
            node = self._leaves.get(path[0], None)
            if node:
                return node
            else:
                raise KeyError(
                    f'Node \"{path}\" not found.'
                )
        elif len(path) > 1:
            branch = path.pop(0)
            subtree = self._branches.get(branch, None)
            if subtree is None:
                raise KeyError(
                    f'Subtree \"{path}\" not found.'
                )
            else:
                return subtree.get_leaf(path)
    
    def get_subtree(self, path: list[str]) -> InheritanceTree:
        """
            Returns a subtree of the leaf node.

            Input:
                path: list[str], path to the subtree node, with the last entry the name of the branch

            Returns:
                InheritanceTree, returns the branch node instance.
        """
        if not self._is_valid:
            self.validate()
        # Make a copy of the path to prevent overrides
        path = copy.deepcopy(path if isinstance(path, list) else list(path))
        if len(path) == 1:
            # Add branch to current level.
            subtree = self._branches.get(path[0], None)
            if subtree is None:
                raise KeyError(
                    f'Subtree \"{path}\" not found.'
                )
        elif len(path) > 1:
            branch = path.pop(0)
            subtree = self._branches.get(branch, None)
            if subtree is None:
                raise KeyError(
                    f'Subtree \"{path}\" not found.'
                )
            return subtree.get_subtree(path) if subtree else None
        
    def to_dict(self,) -> dict:
        """
            InheritanceTree dict serializer.
        """
        if not self._is_valid:
            self.validate()
        # Collect leaves
        return {
            **{l: il.to_dict() for l, il in self._leaves.items()},
            **{b: t.to_dict() for b, t in self._branches.items()}
        }
    
    @classmethod
    def from_dict(cls, d: dict, path: list[str] = []) -> 'InheritanceTree':
        """
            InheritanceTree dict deserializer.
        """
        tree = cls(path)
        for k, v in d.items():
            if isinstance(v, dict):
                if v.get('flags', None) is not None:
                    tree.add_leaf([k], **v)
                else:
                    tree.add_branch([k])
                    tree._branches[k] = cls.from_dict(v, path=path+[k])
            else:
                raise TypeError(
                    f'Expected \"v\" to be a dict, but got \"{v}\".'
                )
        return tree

    @property
    def path(self,) -> list[str]:
        """
            Returns the path of the branch node.
        """
        return self._current_path

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
