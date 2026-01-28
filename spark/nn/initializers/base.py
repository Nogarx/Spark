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
from spark.core.config import BaseSparkConfig
from spark.core.config_validation import TypeValidator

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InitializerConfig(BaseSparkConfig, abc.ABC):
    """
        Base initializers configuration class.
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
        Base (abstract) class for all Spark initializers.
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
            import copy
            self.config = copy.deepcopy(config)
            self.config.merge(partial=kwargs)

    @classmethod 
    def get_config_spec(cls) -> type[InitializerConfig]:
        """
            Returns the default configuration class associated with this module.
        """
        type_hints = tp.get_type_hints(cls)
        return type_hints['config']

    @abc.abstractmethod
    def __call__(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        raise NotImplementedError

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################