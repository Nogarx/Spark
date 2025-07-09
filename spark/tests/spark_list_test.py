
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import jax.numpy as jnp
from jax import Array

# TODO: Fix this test.
from spark.core.payloads import SparkPayload

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# --- Fixtures ---
# Fixtures provide a fixed baseline upon which tests can reliably and repeatedly execute.

@pytest.fixture
def sample_list_data():
    """
        Provides a standard Python list.
    """
    return [1, 2, 3, 4]

@pytest.fixture
def spark_list_float(sample_list_data):
    """
        Provides a SparkPayload instance with float32 dtype.
    """
    return SparkPayload(sample_list_data, dtype=jnp.float32)

@pytest.fixture
def spark_list_int():
    """
        Provides a SparkPayload instance with int32 dtype.
    """
    return SparkPayload([10, 20, 30], dtype=jnp.int32)

@pytest.fixture
def spark_list_scalar():
    """
        Provides a SparkPayload instance initialized from a scalar.
    """
    return SparkPayload(5, dtype=jnp.float32)


# --- Test Cases ---

def test_initialization_from_list(spark_list_float, sample_list_data):
    """
        Tests if the SparkPayload initializes correctly from a list.
    """
    assert isinstance(spark_list_float, SparkPayload)
    assert isinstance(spark_list_float._data, list)
    assert spark_list_float._data == [1.0, 2.0, 3.0, 4.0]
    assert spark_list_float.dtype == jnp.float32

def test_initialization_from_scalar(spark_list_scalar):
    """
        Tests if the SparkPayload initializes correctly from a scalar value.
    """
    assert isinstance(spark_list_scalar, SparkPayload)
    assert isinstance(spark_list_scalar._data, list)
    assert spark_list_scalar._data == [5.0]
    assert spark_list_scalar.dtype == jnp.float32

def test_representation(spark_list_float):
    """
        Tests the __repr__ method to ensure it mimics a JAX array.
    """
    expected_repr = repr(jnp.array([1., 2., 3., 4.], dtype=jnp.float32))
    assert repr(spark_list_float) == expected_repr

def test_array_protocol(spark_list_float):
    """
        Tests the __array__ protocol for JAX/NumPy interoperability.
    """
    jax_array = jnp.array(spark_list_float)
    assert isinstance(jax_array, Array)
    assert jax_array.dtype == jnp.float32
    assert jnp.allclose(jax_array, jnp.array([1., 2., 3., 4.]))

# --- Immutability Tests ---

def test_setitem_immutability(spark_list_float):
    """
        Tests that __setitem__ raises a TypeError.
    """
    with pytest.raises(TypeError, match="SparkPayload objects are immutable."):
        spark_list_float[0] = 100

def test_delitem_immutability(spark_list_float):
    """
        Tests that __delitem__ raises a TypeError.
    """
    with pytest.raises(TypeError, match="SparkPayload objects are immutable."):
        del spark_list_float[0]

# --- Operator Tests ---

def test_unary_operators(spark_list_float):
    """
        Tests unary plus (+) and minus (-).
    """
    assert jnp.allclose(-spark_list_float, jnp.array([-1., -2., -3., -4.]))
    assert jnp.allclose(+spark_list_float, jnp.array([1., 2., 3., 4.]))

def test_addition(spark_list_float):
    """
        Tests addition with scalars, JAX arrays, and other SparkPayloads.
    """
    assert jnp.allclose(spark_list_float + 10, jnp.array([11., 12., 13., 14.]))
    assert jnp.allclose(10 + spark_list_float, jnp.array([11., 12., 13., 14.])) # Reflected
    other_sl = SparkPayload([4, 3, 2, 1])
    assert jnp.allclose(spark_list_float + other_sl, jnp.array([5., 5., 5., 5.]))

def test_multiplication(spark_list_float):
    """
        Tests multiplication with scalars and other arrays.
    """
    assert jnp.allclose(spark_list_float * 3, jnp.array([3., 6., 9., 12.]))
    assert jnp.allclose(3 * spark_list_float, jnp.array([3., 6., 9., 12.])) # Reflected

def test_division(spark_list_float):
    """
        Tests true division.
    """
    assert jnp.allclose(spark_list_float / 2, jnp.array([0.5, 1., 1.5, 2.]))
    assert jnp.allclose(1 / spark_list_float, jnp.array([1., 0.5, 1/3, 0.25])) # Reflected

def test_comparison_operators(spark_list_float):
    """
        Tests ==, !=, <, > operators.
    """
    other_array = jnp.array([1, 5, 2, 6])
    assert jnp.array_equal((spark_list_float == other_array), jnp.array([True, False, False, False]))
    assert jnp.array_equal((spark_list_float < other_array), jnp.array([False, True, False, True]))

# --- JAX-like Property and Method Tests ---

def test_properties(spark_list_float):
    """
        Tests JAX-like properties .shape, .dtype, .T, etc.
    """
    assert spark_list_float.shape == (4,)
    assert spark_list_float.dtype == jnp.float32
    assert spark_list_float.ndim == 1
    assert spark_list_float.size == 4
    # For a 1D array, .T is a no-op but should still work
    assert jnp.allclose(spark_list_float.T, jnp.array([1., 2., 3., 4.]))

def test_methods(spark_list_float):
    """
        Tests forwarded JAX-like methods like .sum() and .mean().
    """
    assert spark_list_float.sum() == 10.0
    assert spark_list_float.mean() == 2.5
    reshaped = spark_list_float.reshape((2, 2))
    assert reshaped.shape == (2, 2)
    assert isinstance(reshaped, Array) # Ensure methods return JAX arrays

def test_indexing(spark_list_float):
    """
        Tests if indexing and slicing work correctly and return JAX objects.
    """
    # Single item
    item = spark_list_float[1]
    assert isinstance(item, Array)
    assert item.shape == () # It's a 0-D array (scalar)
    assert item == 2.0
    
    # Slice
    slice_ = spark_list_float[1:3]
    assert isinstance(slice_, Array)
    assert slice_.shape == (2,)
    assert jnp.allclose(slice_, jnp.array([2., 3.]))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################