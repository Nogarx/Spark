
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import jax 
import jax.numpy as jnp
import numpy as np
import typing as tp
import flax.nnx as nnx
import spark
np.random.seed(42)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

data_test = [
    # Input interfaces
    (
        'ConstantInitializer_1',
        spark.nn.initializers.ConstantInitializer, 
        {'key': jax.random.key(42), 'shape': (2,3,4,5)},
        {'scale':8, 'dtype':jnp.uint8}
    ),
    (
        'ConstantInitializer_2',
        spark.nn.initializers.ConstantInitializer, 
        {'key': jax.random.key(42), 'shape': (8,2,4)},
        {'scale':8, 'dtype':jnp.float16}
    ),
    (
        'UniformInitializer_1',
        spark.nn.initializers.UniformInitializer, 
        {'key': jax.random.key(42), 'shape': (2,3,4,5)},
        {'scale':8, 'dtype':jnp.int16, 'max_value':4}
    ),
    (
        'UniformInitializer_2',
        spark.nn.initializers.UniformInitializer, 
        {'key': jax.random.key(42), 'shape': (8,2,4)},
        {'scale':8, 'dtype':jnp.float32, 'max_value':4, 'min_value':2}
    ),
    (
        'SparseUniformInitializer_1',
        spark.nn.initializers.SparseUniformInitializer, 
        {'key': jax.random.key(42), 'shape': (2,3,4,5)},
        {'scale':4, 'dtype':jnp.uint32, 'density': 1.0}
    ),
    (
        'SparseUniformInitializer_2',
        spark.nn.initializers.SparseUniformInitializer, 
        {'key': jax.random.key(42), 'shape': (8,2,4)},
        {'scale':4, 'dtype':jnp.float16, 'density': 0.5}
    ),
    (
        'NormalizedSparseUniformInitializer_1',
        spark.nn.initializers.NormalizedSparseUniformInitializer, 
        {'key': jax.random.key(42), 'shape': (10,5), 'norm_axes': (0,)},
        {'scale':4, 'dtype':jnp.float16, 'density': 1.0}
    ),
    (
        'NormalizedSparseUniformInitializer_2',
        spark.nn.initializers.NormalizedSparseUniformInitializer, 
        {'key': jax.random.key(42), 'shape': (2,3,4,5), 'norm_axes': (1,3)},
        {'scale':4, 'dtype':jnp.float32, 'density': 1.0}
    ),
    (
        'NormalizedSparseUniformInitializer_3',
        spark.nn.initializers.NormalizedSparseUniformInitializer, 
        {'key': jax.random.key(42), 'shape': (8,2,4), 'norm_axes': (0,)},
        {'scale':4, 'dtype':jnp.float16, 'density': 0.5}
    ),
]

relaxed_min_test = (
    spark.nn.initializers.SparseUniformInitializer,
    spark.nn.initializers.NormalizedSparseUniformInitializer,
)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.mark.parametrize('name, initializer_cls, initializer_input, initializer_config_kwargs', data_test)
def test_initializer(
        name: str,
        initializer_cls: type[spark.nn.initializers.Initializer], 
        initializer_input: dict[str, spark.SparkPayload], 
        initializer_config_kwargs: dict[str, tp.Any]
    ) -> None:
    """
        Performs a simple initializer validation.
    """
    # Extract norm_axes if present
    norm_axes = initializer_input.get('norm_axes', None)
    if norm_axes:
        initializer_config_kwargs['norm_axes'] = norm_axes
        initializer_input.pop('norm_axes')
    # Build initializer and output array
    initializer = initializer_cls(**initializer_config_kwargs)
    array: jax.Array = initializer(**initializer_input)
    # Basic validation.
    assert array.shape == initializer_input['shape']
    assert array.dtype == initializer_config_kwargs['dtype']
    # Validate min, max.
    min_value = initializer_config_kwargs.get('min_value', None)
    max_value = initializer_config_kwargs.get('max_value', None)
    if min_value and issubclass(initializer_cls, relaxed_min_test):
        assert jnp.all(jnp.where(array != 0, array >= min_value, True))
    elif min_value:
        assert jnp.all(array >= min_value)
    if max_value:
        assert jnp.all(array <= max_value)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################