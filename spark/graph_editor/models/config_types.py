#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import enum
import types
import typing as tp
import importlib
import numpy as np

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# NOTE: SparkConfigMeta promotes every annotated attribute into a dataclass field and stores the parsed
# annotation under the "valid_types" metadata entry. That entry holds either real types (e.g. <class 'float'>)
# or the raw annotation string (e.g. 'float | jax.Array | Initializer'), depending on whether the declaring
# module uses "from __future__ import annotations". Every candidate is reduced to a set of normalized string
# tokens before being classified.

class FieldKind(enum.Enum):
    """
        Editor-level classification of a configuration field.
    """
    BOOL = enum.auto()
    INT = enum.auto()
    FLOAT = enum.auto()
    STR = enum.auto()
    DTYPE = enum.auto()
    INT_TUPLE = enum.auto()
    FLOAT_TUPLE = enum.auto()
    ARRAY = enum.auto()
    MODULE_SPECS = enum.auto()
    UNKNOWN = enum.auto()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Type aliases used across the framework, resolved lazily and expanded into their underlying types.
_ALIAS_SOURCES: dict[str, tuple[str, str]] = {
    'PlasticityParamLike': ('spark.nn.components.plasticity.base', 'PlasticityParamLike'),
    'DTypeLike': ('jax.typing', 'DTypeLike'),
    'ArrayLike': ('jax.typing', 'ArrayLike'),
}

_ALIAS_CACHE: dict[str, frozenset[str]] = {}

_ARRAY_TOKENS = frozenset({'jax.Array', 'Array', 'ndarray', 'numpy.ndarray', 'ArrayLike'})
_DTYPE_TOKENS = frozenset({'DTypeLike', 'SupportsDType'})
_INITIALIZER_TOKENS = frozenset({'Initializer', 'InitializerConfig', 'MaskedInitializer'})

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _canonical(text: str) -> str:
    """
        Normalizes a type string so that equivalent annotations produce identical tokens.
    """
    return ''.join(text.split())

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _resolve_alias(name: str) -> frozenset[str]:
    """
        Expands a known type alias into its underlying tokens.
    """
    if name in _ALIAS_CACHE:
        return _ALIAS_CACHE[name]
    source = _ALIAS_SOURCES.get(name, None)
    if source is None:
        return frozenset()
    # Guard against self-referencing aliases.
    _ALIAS_CACHE[name] = frozenset()
    try:
        module = importlib.import_module(source[0])
        tokens = _tokens_from(getattr(module, source[1]))
    except Exception:
        tokens = set()
    _ALIAS_CACHE[name] = frozenset(tokens)
    return _ALIAS_CACHE[name]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _tokens_from_str(text: str) -> set[str]:
    """
        Tokenizes a single (non-union) annotation string.
    """
    raw = text.strip()
    if not raw:
        return set()
    tokens = {_canonical(raw)}
    # Expose the unqualified name for dotted paths (e.g. "jax.Array" -> "Array").
    canonical = _canonical(raw)
    if '.' in canonical and '[' not in canonical:
        tokens.add(canonical.split('.')[-1])
    if raw in ('None', 'NoneType'):
        tokens.add('None')
    tokens |= _resolve_alias(raw)
    return tokens

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _tokens_from(candidate: tp.Any) -> set[str]:
    """
        Reduces a single type candidate (string, type or typing construct) to a set of normalized tokens.
    """
    if candidate is None:
        return set()
    if isinstance(candidate, str):
        tokens = set()
        for part in candidate.split('|'):
            tokens |= _tokens_from_str(part)
        return tokens
    if candidate is type(None):
        return {'None'}
    origin = tp.get_origin(candidate)
    # Unions
    if origin is tp.Union or origin is types.UnionType:
        tokens = set()
        for arg in tp.get_args(candidate):
            tokens |= _tokens_from(arg)
        return tokens
    # Parametrized generics (tuple[int, ...], type[Any], ...)
    if origin is not None:
        tokens = {_canonical(str(candidate))}
        origin_name = getattr(origin, '__name__', None)
        if origin_name:
            tokens.add(origin_name)
        return tokens
    # Plain classes
    if isinstance(candidate, type):
        tokens = {candidate.__name__}
        module = getattr(candidate, '__module__', '')
        if module:
            tokens.add(f'{module.split(".")[0]}.{candidate.__name__}')
        tokens |= _resolve_alias(candidate.__name__)
        return tokens
    return {_canonical(str(candidate))}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def type_tokens(type_hint: tp.Any = None, valid_types: tp.Any = None) -> frozenset[str]:
    """
        Builds the normalized token set describing a configuration field.

        Input:
            type_hint: tp.Any, the dataclass field annotation (may be a string).
            valid_types: tp.Any, the "valid_types" entry of the field metadata.

        Returns:
            frozenset[str], normalized type tokens.
    """
    tokens: set[str] = set()
    if valid_types is not None:
        candidates = valid_types if isinstance(valid_types, (list, tuple, set, frozenset)) else [valid_types]
        for candidate in candidates:
            tokens |= _tokens_from(candidate)
    if type_hint is not None:
        tokens |= _tokens_from(type_hint)
    return frozenset(tokens)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _has_prefix(tokens: frozenset[str], prefix: str) -> bool:
    return any(token.startswith(prefix) for token in tokens)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def classify(type_hint: tp.Any = None, valid_types: tp.Any = None, tokens: frozenset[str] | None = None) -> FieldKind:
    """
        Maps a configuration field to the editor widget family that can safely edit it.
    """
    tokens = type_tokens(type_hint, valid_types) if tokens is None else tokens
    if _has_prefix(tokens, 'tuple[ModuleSpecs'):
        return FieldKind.MODULE_SPECS
    if tokens & _DTYPE_TOKENS:
        return FieldKind.DTYPE
    if _has_prefix(tokens, 'tuple[int'):
        return FieldKind.INT_TUPLE
    if 'bool' in tokens:
        return FieldKind.BOOL
    if 'float' in tokens:
        return FieldKind.FLOAT
    if 'int' in tokens:
        return FieldKind.INT
    if _has_prefix(tokens, 'tuple[float'):
        return FieldKind.FLOAT_TUPLE
    if 'str' in tokens:
        return FieldKind.STR
    if tokens & _ARRAY_TOKENS:
        return FieldKind.ARRAY
    return FieldKind.UNKNOWN

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_optional(tokens: frozenset[str]) -> bool:
    """
        Returns True if the field explicitly accepts None.
    """
    return 'None' in tokens

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def accepts_array(tokens: frozenset[str]) -> bool:
    """
        Returns True if the field accepts a raw array value.
    """
    return bool(tokens & _ARRAY_TOKENS)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def accepts_initializer(tokens: frozenset[str]) -> bool:
    """
        Returns True if the field explicitly accepts an Initializer.
    """
    return bool(tokens & _INITIALIZER_TOKENS)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def initializer_policy(tokens: frozenset[str], metadata: dict | None = None) -> tuple[bool, bool]:
    """
        Decides whether a field may (and must) be defined through an initializer.

        Input:
            tokens: frozenset[str], normalized type tokens of the field.
            metadata: dict, field metadata.

        Returns:
            tuple[bool, bool], (an initializer is allowed, an initializer is mandatory).
    """
    metadata = metadata or {}
    if not metadata.get('allows_init', False):
        return (False, False)
    allowed = accepts_initializer(tokens) or not is_optional(tokens)
    required = allowed and classify(tokens=tokens) is FieldKind.ARRAY
    return (allowed, required)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def type_label(tokens: frozenset[str], type_hint: tp.Any = None) -> str:
    """
        Human readable representation of a field type, used for tooltips.
    """
    if isinstance(type_hint, str) and type_hint:
        return type_hint
    if type_hint is not None and not isinstance(type_hint, str):
        origin = tp.get_origin(type_hint)
        if origin is None and isinstance(type_hint, type):
            return type_hint.__name__
        return str(type_hint)
    return ' | '.join(sorted(tokens)) if tokens else 'unknown'

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def dtype_key(value: tp.Any) -> str | None:
    """
        Canonical name of a dtype-like value.
        """
    if value is None:
        return None
    try:
        return np.dtype(value).name
    except Exception:
        name = getattr(value, '__name__', None)
        return str(name) if name else str(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def values_equal(first: tp.Any, second: tp.Any) -> bool:
    """
        Comparison that tolerates arrays, dtypes and objects with an ambiguous __eq__.
    """
    if first is second:
        return True
    if first is None or second is None:
        return first is None and second is None
    # Arrays (jax/numpy) return element-wise results, which cannot be used as a boolean.
    if hasattr(first, 'shape') or hasattr(second, 'shape'):
        try:
            return bool(np.array_equal(np.asarray(first), np.asarray(second)))
        except Exception:
            return False
    try:
        return bool(first == second)
    except Exception:
        return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def coerce(kind: FieldKind, value: tp.Any) -> tp.Any:
    """
        Casts a widget value to the python type expected by the configuration field.
    """
    if value is None:
        return None
    try:
        if kind is FieldKind.BOOL:
            return bool(value)
        if kind is FieldKind.INT:
            return int(round(float(value)))
        if kind is FieldKind.FLOAT:
            return float(value)
        if kind is FieldKind.INT_TUPLE:
            return tuple(int(v) for v in value)
        if kind is FieldKind.FLOAT_TUPLE:
            return tuple(float(v) for v in value)
        if kind is FieldKind.STR:
            return str(value)
    except (TypeError, ValueError):
        return value
    return value

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
