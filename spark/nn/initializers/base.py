#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp 
import inspect
import typing as tp
import abc
import dataclasses as dc
from jax.typing import DTypeLike
from spark.core.config import SparkConfig
from spark.core.config_validation import TypeValidator
from spark.core.backend import data as set_data_fn

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InitializerConfig(SparkConfig, abc.ABC):
    """
        Base configuration for initializers.

        Parameters
        ----------
        dtype : DTypeLike, default jnp.float16
            Dtype of the produced array.
        scale : int or float, default 1
            Factor applied to the produced array.
        min_value : int or float or None, default None
            Lower bound. Applied by clipping in the initializers of this package.
        max_value : int or float or None, default None
            Upper bound. Applied by clipping in the initializers of this package.
    """

    # NOTE: x64 dtypes require manual override and it is unlikely that they are going to be required anyway; similar with complex numbers.
    dtype: DTypeLike = dc.field(
        default = jnp.float16, 
        metadata = {
            'validators': [
                TypeValidator,
            ], 
            'value_options': [
                jnp.uint8,
                jnp.uint16,
                jnp.uint32,
                jnp.int8,
                jnp.int16,
                jnp.int32,
                jnp.float16,
                jnp.float32,
            ],
            'description': 'Final dtype for the output jax.Array.',
        }
    )
    scale: int | float = dc.field(
        default = 1, 
        metadata = {
            'validators': [
            ], 
            'description': 'Scale factor for the jax.Array.',
        }
    )
    min_value: int | float | None = dc.field(
        default = None, 
        metadata = {
            'validators': [
            ], 
            'description': 'Min value for the jax.Array. Note that some initializers implement this as a clipping value.',
        }
    )
    max_value: int | float | None = dc.field(
        default = None, 
        metadata = {
            'validators': [
            ], 
            'description': 'Max value for the jax.Array. Note that some initializers implement this as a clipping value.',
        }
    )

ConfigT = tp.TypeVar("ConfigT", bound=InitializerConfig)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class Initializer(abc.ABC):
    """
        Base class for initializers.

        An initializer produces the array a parameter starts from. Passing one in place of a value
        lets a configuration describe a whole pool without holding the array: the array is drawn
        at build time, once the shape is known.

        A subclass must declare its configuration through the ``config`` annotation, which is also
        what it falls back to when constructed without one.

        Parameters
        ----------
        config : InitializerConfig, optional
            Initializer configuration. Its fields may also be given as keyword arguments.

        See Also
        --------
        ConstantInitializer : Every entry the same value.
        UniformInitializer : Entries drawn uniformly.
        SparseUniformInitializer : Uniform entries with a fraction zeroed.
    """
    config: InitializerConfig
    default_config: type[ConfigT]

    # NOTE: Similar idea to SparkModule and SparkConfig, to force an Initializer to define a default config. 
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Special cases and abstract classes dont need config.
        is_abc = inspect.isabstract(cls) and len(getattr(cls, '__abstractmethods__', set())) == 0
        if is_abc:
            return
        # Check if defines config
        resolved_hints = tp.get_type_hints(cls)
        config_type = resolved_hints.get('config')
        if not config_type or not issubclass(config_type, InitializerConfig):
            raise AttributeError('Initializer must define a valid config: type[InitializerConfig] attribute.')
        cls.default_config = tp.cast(type[ConfigT], config_type)

    def __init__(self, *, config: ConfigT | None = None, **kwargs) -> None:
        # Override config if provided
        if config is None:
            self.config = self.default_config(**kwargs)
        else:
            self.config = config.merge(**kwargs)

    @classmethod 
    def get_config_spec(cls) -> type[InitializerConfig]:
        """
            Returns the default configuration class associated with this module.
        """
        type_hints = tp.get_type_hints(cls)
        return type_hints['config']

    @abc.abstractmethod
    def __call__(self, key: jax.Array, shape: tuple[int, ...], **kwargs) -> jax.Array:
        """
            Draws the array.

            Parameters
            ----------
            key : jax.Array
                PRNG key.
            shape : tuple of int
                Shape of the array to draw.
            **kwargs
                Extra arguments accepted by the concrete initializer.

            Returns
            -------
            jax.Array
                The drawn array, cast to ``dtype``.
        """
        raise NotImplementedError

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class MaskedInitializer(abc.ABC):
    """
        Base class for initializers that take a mask.

        Called with a mask alongside the key and the shape, which is what lets an initializer draw
        different values for different groups of entries.
    """

    @abc.abstractmethod
    def __call__(self, mask: jax.Array, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """
            Draws the array under a mask.

            Parameters
            ----------
            mask : jax.Array
                Selects which entries are drawn together.
            key : jax.Array
                PRNG key.
            shape : tuple of int
                Shape of the array to draw.

            Returns
            -------
            jax.Array
                The drawn array.
        """
        raise NotImplementedError

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################