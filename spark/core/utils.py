#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import re
import jax
import jax.numpy as jnp
import enum
import string
import numpy as np
import typing as tp
import collections.abc
import copy 
import dataclasses as dc
from math import prod 
from collections import defaultdict
from collections.abc import MutableMapping

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def normalize_str(s: str) -> str:
    """
        Converts a string to lowercase snake_case.

        Parameters
        ----------
        s : str
            String to normalize.

        Returns
        -------
        str
            The normalized string.
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

def to_human_readable(s: str, capitalize_all: bool = True) -> str:
    """
        Converts a string from a programming case into a readable one.

        Parameters
        ----------
        s : str
            String to convert.
        capitalize_all : bool, default False
            Title-case every word instead of the first one alone.

        Returns
        -------
        str
            The readable string.
    """

    def _looks_like_acronym(w: str) -> bool:
        return w.isupper() or any(c.isupper() for c in w[1:])

    # Sanity check
    if not isinstance(s, str) or not s:
        raise TypeError('\"s\" must be a non-empty string.')

    # Last capital of a run starts the new word.
    s = re.compile(r'([A-Z]+)([A-Z][a-z])').sub(r'\1_\2', s)
    # Digits absorb into the preceding token.
    s = re.compile(r'([0-9])([A-Z][a-z])').sub(r'\1_\2', s)
    # Separate words
    s = re.compile(r'([a-z])([A-Z])').sub(r'\1_\2', s)
    words = [w for w in re.compile(r'[-_\s]+').split(s) if w]

    if capitalize_all:
        # Uppercase the first char only; never touch the tail, so RD stays RD.
        return ' '.join(w[:1].upper() + w[1:] for w in words)

    head, *tail = words
    return ' '.join([head[:1].upper() + head[1:]]
                    + [w if _looks_like_acronym(w) else w.lower() for w in tail])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_einsum_labels(num_dims: int, offset: int = 0) -> str:
    """
        Builds a run of einsum labels.

        Parameters
        ----------
        num_dims : int
            Number of labels to generate.
        offset : int, default 0
            Number of labels to skip before the first one.

        Returns
        -------
        str
            ``num_dims`` distinct labels, starting after ``offset``.
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
        Builds the einsum labels naming a given set of axes.

        Parameters
        ----------
        axes : tuple of int
            Axis indices to name.
        ignore_repeated : bool, default False
            Allow the same axis to appear more than once.

        Returns
        -------
        str
            One label per entry of ``axes``, in the order given.

        Raises
        ------
        ValueError
            If an axis is negative, beyond the number of available labels, or repeated while
            ``ignore_repeated`` is False.
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
        Builds the einsum string of a generalized dot product.

        The shared axes are contracted and the remaining ones are kept::

            right:  (c,d)*(a,b,c,d)=(a,b)  cd,abcd->ab    (a,b,c,d)*(c,d)=(a,b)  abcd,cd->ab
            left:   (a,b)*(a,b,c,d)=(c,d)  ab,abcd->cd    (a,b,c,d)*(a,b)=(c,d)  abcd,ab->cd

        Parameters
        ----------
        x : tuple of int
            Shape of the first operand.
        y : tuple of int
            Shape of the second operand.
        ignore_one_dims : bool, default True
            Drop the axes of length one before building the labels.
        side : {'right', 'left'}, default 'right'
            Which end of the longer shape the shorter one aligns with.

        Returns
        -------
        str
            The einsum string.
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
        Builds the einsum string of a generalized dot product that keeps the shared axes.

        The shared axes are reduced onto themselves rather than contracted away::

            right:  (a,b)*(a,b,c,d)=(a,b)  ab,abcd->ab    (a,b,c,d)*(a,b)=(a,b)  abcd,ab->ab
            left:   (c,d)*(a,b,c,d)=(c,d)  cd,abcd->cd    (a,b,c,d)*(c,d)=(c,d)  abcd,cd->cd

        Parameters
        ----------
        x : tuple of int
            Shape of the first operand.
        y : tuple of int
            Shape of the second operand.
        ignore_one_dims : bool, default True
            Drop the axes of length one before building the labels.
        side : {'right', 'left'}, default 'right'
            Which end of the longer shape the shorter one aligns with.

        Returns
        -------
        str
            The einsum string.
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

def get_einsum_dot_exp_string(x: tuple[int, ...], y: tuple[int, ...], ignore_one_dims: bool = True, side: str = 'right') -> str:
    """
        Builds the einsum string of a generalized outer product.

        Nothing is contracted; the result carries the axes of both operands::

            right:  (a,b)*(a,b,c,d)=(a,b,c,d)  ab,abcd->abcd
            left:   (c,d)*(a,b,c,d)=(a,b,c,d)  cd,abcd->abcd
            none:   (a,b)*(c,d)=(a,b,c,d)      ab,cd->abcd
            flip:   (a,b)*(c,d)=(c,d,a,b)      cd,ab->abcd

        Parameters
        ----------
        x : tuple of int
            Shape of the first operand.
        y : tuple of int
            Shape of the second operand.
        ignore_one_dims : bool, default True
            Drop the axes of length one before building the labels.
        side : {'right', 'left', 'none', 'flip'}, default 'right'
            How the axes of the two operands are laid out in the result.

        Returns
        -------
        str
            The einsum string.
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
    elif side == 'flip' or side == 'f':
        y_indices = get_einsum_labels(len(x), offset=0)
        x_indices = get_einsum_labels(len(y), offset=len(x))
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
        Checks that an object reads as a shape and returns it.

        Parameters
        ----------
        obj : Any
            Object to read as a shape.

        Returns
        -------
        tuple of int
            The shape.

        Raises
        ------
        TypeError
            If the object does not read as a tuple of integers.
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
        Checks that an object reads as a list of shapes and returns it.

        Parameters
        ----------
        obj : Any
            Object to read as a list of shapes.

        Returns
        -------
        list of tuple of int
            The shapes.

        Raises
        ------
        TypeError
            If the object does not read as a list of tuples of integers.
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

def merge_shape_list(shape_list: list[tuple[int, ...]]) -> tuple[int, ...]:
    """
        Merges a list of shapes into one.

        Parameters
        ----------
        shape_list : list of tuple of int
            Shapes to merge.

        Returns
        -------
        tuple of int
            The merged shape.
    """
    shape_list = validate_list_shape(shape_list)
    return tuple([sum([prod(s) for s in shape_list])])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def contract_axes(array: tp.Any, axes: tuple[int, ...], shape: tuple[int, ...]) -> tuple[tp.Any, bool]:
    """
        Contracts an array over the given axes when it is constant along them.

        Parameters
        ----------
        array : Any
            Array to contract.
        axes : tuple of int
            Axes to contract over.
        shape : tuple of int
            Shape the array is read against.

        Returns
        -------
        array : Any
            The contracted array, or the original one when the contraction does not apply.
        contracted : bool
            Whether the array was contracted.
    """
    if not isinstance(array, jax.Array) or len(axes) == 0:
        return array, True
    aligned = array.reshape((1,) * (len(shape) - array.ndim) + array.shape)
    reduced = aligned
    for axis in axes:
        reduced = jax.lax.index_in_dim(reduced, 0, axis=axis, keepdims=True)
    try:
        is_constant = bool(jnp.all(aligned == reduced))
    except Exception:
        return array, False
    return (reduced, True) if is_constant else (array, False)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def contracted_shape(shape: tuple[int, ...], axes: tuple[int, ...]) -> tuple[int, ...]:
    """
        Returns the shape `contract_axes` produces.

        Parameters
        ----------
        shape : tuple of int
            Shape before the contraction.
        axes : tuple of int
            Axes contracted over.

        Returns
        -------
        tuple of int
            The shape with the contracted axes dropped.
    """
    return tuple(1 if axis in axes else size for axis, size in enumerate(shape))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_shape(obj: tp.Any) -> bool:
    """
        Whether an object reads as a shape.

        Parameters
        ----------
        obj : Any
            Object to check.

        Returns
        -------
        bool
    """
    try: 
        validate_shape(obj)
        return True
    except:
        return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_list_shape(obj: tp.Any) -> bool:
    """
        Whether an object reads as a list of shapes.

        Parameters
        ----------
        obj : Any
            Object to check.

        Returns
        -------
        bool
    """
    try: 
        validate_list_shape(obj)
        return True
    except:
        return False


#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_dict_of(obj: tp.Any, value_cls: type[tp.Any], key_cls: type[tp.Any] = str) -> bool:
    """
        Whether an object is a ``dict[key_cls, value_cls]``.

        Parameters
        ----------
        obj : Any
            Object to check.
        value_cls : type
            Class the values are checked against.
        key_cls : type, default str
            Class the keys are checked against.

        Returns
        -------
        bool
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
        Whether an object is a ``list[cls]``.

        Parameters
        ----------
        obj : Any
            Object to check.
        cls : type
            Class the entries are checked against.

        Returns
        -------
        bool
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
        Whether an object is a dtype.

        Parameters
        ----------
        obj : Any
            Object to check.

        Returns
        -------
        bool
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
        Whether an object is a floating point dtype.

        Parameters
        ----------
        obj : Any
            Object to check.

        Returns
        -------
        bool
            True for a real floating dtype, False for anything else, including integer and
            boolean dtypes.
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
        Renders indented text as an ASCII tree.

        Parameters
        ----------
        text : str
            Lines whose depth is given by their leading spaces.

        Returns
        -------
        str
            The tree.
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

# NOTE: This is just a convinience class to simplify some code inside controllers and is equivalent to two nested dictionaries. 
# Notably, this class produces the same XLA code as using nested dictionaries (after JIT). It's only purpose is to simply notaion.

_K1 = tp.TypeVar('_K1')
_K2 = tp.TypeVar('_K2')
_VT = tp.TypeVar('_VT')

@jax.tree_util.register_pytree_with_keys_class
@dc.dataclass(init=False, eq=False)
class TwoKeyDict(MutableMapping[tuple[_K1, _K2], _VT], tp.Generic[_K1, _K2, _VT]):

    """
        Mapping addressed by a pair of keys.

        ``d[k1, k2]`` reads one entry and ``d[k1]`` reads the inner mapping under the first key.
        Used for the values a graph addresses by (module name, port name).

        Parameters
        ----------
        data : dict of K1 to dict of K2 to VT, optional
            Initial contents, as nested mappings.
    """
    def __init__(self, data: dict[_K1, dict[_K2, _VT]] | None = None) -> None:
        self._data = defaultdict(dict)
        if not data is None:
            for k, v in data.items():
                self._data[k] = v

    @tp.overload
    def __getitem__(self, keys: tuple[_K1, _K2] )-> _VT: ...
    @tp.overload
    def __getitem__(self, keys: _K1)-> dict[_K2, _VT]: ...
    def __getitem__(self, keys):
        if isinstance(keys, tuple):
            k1, k2 = keys
            if k1 not in self._data or k2 not in self._data[k1]:
                raise KeyError(f'Invalid key pair: {keys}')
            return self._data[k1][k2]
        else:
            if keys not in self._data:
                raise KeyError(f'Invalid key: {keys}')
            return self._data[keys]

    @tp.overload
    def __setitem__(self, keys: _K1, value: dict[_K2, _VT]) -> None: ...
    @tp.overload
    def __setitem__(self, keys: tuple[_K1, _K2], value: _VT) -> None: ...
    def __setitem__(self, keys, value) -> None:
        if isinstance(keys, tuple):
            k1, k2 = keys
            self._data[k1][k2] = value
        else:
            self._data[keys] = value

    @tp.overload
    def __delitem__(self, keys: _K1) -> None: ...
    @tp.overload
    def __delitem__(self, keys: tuple[_K1, _K2]) -> None: ...
    def __delitem__(self, keys) -> None:
        if isinstance(keys, tuple):
            k1, k2 = keys
            if k1 not in self._data or k2 not in self._data[k1]:
                raise KeyError(f'Invalid key pair: {keys}')
            del self._data[k1][k2]
        else:
            if keys not in self._data:
                raise KeyError(f'Invalid key: {keys}')
            del self._data[keys]
        
    def __len__(self,) -> int:
        return sum([len(v) for v in self._data.values()])

    def __iter__(self,) -> tp.Iterator[tuple[_K1, _K2]]:
        for key1, subdict in self._data.items():
            for key2 in subdict.keys():
                yield (key1, key2)

    def __str__(self,) -> str:
        _str = []
        for key1, subdict in self._data.items():
            _substr = []
            for key2, value in subdict.items():
                _substr_key = f'\'{key2}\'' if isinstance(key2, str) else str(key2)
                _substr += [f'{_substr_key}: {str(value)}']
            _substr = f'{{{', '.join(_substr)}}}' 
            _str_key = f'\'{key1}\'' if isinstance(key1, str) else str(key1)
            _str += [f'{_str_key}: {_substr}']
        _str = f'{{{', '.join(_str)}}}'
        return _str
    
    def __repr__(self,) -> str:
        _str = []
        for key1, subdict in self._data.items():
            for key2, value in subdict.items():
                _str_key = f'\'{key1}\'' if isinstance(key1, str) else str(key1)
                _substr_key = f'\'{key2}\'' if isinstance(key2, str) else str(key2)
                _str += [f'{_str_key}/{_substr_key}: {value.__repr__()}']
        _str = f'{{{', '.join(_str)}}}'
        return _str

    @tp.overload
    def __contains__(self, keys: _K1) -> bool: ...
    @tp.overload
    def __contains__(self, keys: tuple[_K1, _K2]) -> bool: ...
    def __contains__(self, keys) -> bool:
        if isinstance(keys, tuple):
            k1, k2 = keys
            if k1 not in self._data or k2 not in self._data[k1]:
                return False
            return True
        else:
            if keys not in self._data:
                return False
            return True

    def keys(self) -> tp.KeysView[tp.Tuple[_K1, _K2]]:
        return super().keys()

    def values(self) -> tp.ValuesView[_VT]:
        return super().values()

    def items(self) -> tp.ItemsView[tp.Tuple[_K1, _K2], _VT]:
        return super().items()

    def _ordered_keys(self) -> list[_K1]:
        """
            First level keys, in a deterministic order.
        """
        try:
            return sorted(self._data.keys())
        except TypeError:
            return list(self._data.keys())

    def tree_flatten(self) -> tuple[tuple, tuple]:
        keys = self._ordered_keys()
        return tuple(self._data[key] for key in keys), tuple(keys)

    def tree_flatten_with_keys(self) -> tuple[list, tuple]:
        keys = self._ordered_keys()
        return [(jax.tree_util.DictKey(key), self._data[key]) for key in keys], tuple(keys)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> tp.Self:
        return cls(dict(zip(aux_data, children)))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################