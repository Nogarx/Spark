#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import numpy as np
import jax.numpy as jnp
import typing as tp
import functools
from numpy.typing import ArrayLike, DTypeLike
from jax.typing import ArrayLike as JaxArrayLike

from spark.core.typing import (
    is_dtype_like, is_array_like, is_object_of_type, enforce_annotations
)

from spark.core.payloads import SparkPayload, SpikeArray
from spark.core.config import BaseSparkConfig, SparkConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@pytest.mark.parametrize('dtype_input, expected', [
    # Valid dtypes
    (np.float32, True),
    (jnp.float32, True),
    ('int64', True),
    (int, True),
    (bool, True),
    (np.dtype('float64'), True),
    # Invalid dtypes
    ('not-a-type', False),
    (123, False),
    ([1, 2], False),
])

def test_is_dtype_like_exhaustive(dtype_input: tp.Any, expected: bool) -> None:
    assert is_dtype_like(dtype_input) == expected

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_is_array_like_variants() -> None:
    # Valid arrays
    assert is_array_like(np.zeros((2, 2))) is True
    assert is_array_like(jnp.ones((3,))) is True
    # Invalid arrays
    assert is_array_like([1, 2, 3]) is False 
    assert is_array_like(5.0) is False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestRecursiveTypeChecker:
    
    @pytest.mark.parametrize('obj, _type, expected', [
        # Nesting
        ([[1, 2], [3, 4]], list[list[int]], True),
        ([[1, 2], [3, '4']], list[list[int]], False),
        # Optional & Union Mix
        (None, tp.Optional[list[int]], True),
        ([1, 2], tp.Union[int, list[int]], True),
        (3.14, tp.Union[int, list[int]], False),
        # Mappings
        ({'a': [1, 2], 'b': [3, 4]}, dict[str, list[int]], True),
        ({'a': [1, '2']}, dict[str, list[int]], False),
        # Tuple
        ((1, 'a', 3.0), tuple[int, str, float], True),
        ((1, 2, 3, 4), tuple[int, ...], True),
        ((1, 2, '3'), tuple[int, ...], False),
        # Any
        ({'a': [1, 2, 3]}, tp.Any, True),
        (lambda x: x, tp.Any, True),
    ])

    def test_nested_structures(self, obj: tp.Any, _type: type, expected: bool) -> None:
        assert is_object_of_type(obj, _type) == expected

    def test_callable_signature_matching(self) -> None:
        # Correct signature
        def func_a(x: int, y: str) -> bool: return True
        assert is_object_of_type(func_a, tp.Callable[[int, str], bool]) is True
        # Wrong return type hint
        def func_b(x: int) -> int: return x
        assert is_object_of_type(func_b, tp.Callable[[int], str]) is False
        # Wrong parameter count
        assert is_object_of_type(func_b, tp.Callable[[int, int], int]) is False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestEnforcerExhaustive:

    def test_class_method_handling(self) -> None:
        class MockModel:
            @enforce_annotations
            def predict(self, data: np.ndarray, threshold: float = 0.5):
                return True
        model = MockModel()
        # Should pass
        assert model.predict(np.array([1]), 0.7) is True
        # Should fail
        with pytest.raises(TypeError, match='Positional argument "data"'):
            model.predict([1, 2, 3], 0.5)

    def test_jax_array_like_annotation(self) -> None:
        @enforce_annotations
        def compute(x: JaxArrayLike):
            return x
        # Pass with Jax/Numpy array
        assert compute(jnp.array([1.0])) is not None
        assert compute(np.array([1.0])) is not None
        # Fail with list
        with pytest.raises(TypeError, match='expected ArrayLike'):
            compute([1.0, 2.0])

    def test_keyword_argument_validation(self) -> None:
        @functools.partial(enforce_annotations, validate_keys=True)
        def solver(*, alpha: float, beta: int) -> float:
            return alpha + beta
        # Correct kwargs
        assert solver(alpha=0.1, beta=1) == 1.1
        # Unexpected kwarg
        with pytest.raises(TypeError, match='not part of the function signature'):
            solver(alpha=0.1, beta=1, gamma=3)

    def test_undefined_behavior_runtime_error(self):
        """Force the 'Undefined behaviour' RuntimeError for tuples."""
        @enforce_annotations
        def bad_tuple_func(x: tuple[int, int]):
            return x
        with pytest.raises(RuntimeError, match='Undefined behaviour'):
            bad_tuple_func((1, 2, 3))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_enforce_annotations_preserves_metadata():
    def target_func(x: int):
        """Original Docstring"""
        return x

    wrapped = enforce_annotations(target_func)
    assert wrapped.__name__ == target_func.__name__
    assert wrapped.__doc__ == target_func.__doc__

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_spark_obj():
    @enforce_annotations
    def target_func(x: SpikeArray) -> SpikeArray:
        return x
    assert target_func(SpikeArray([1])) == SpikeArray([1])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_spark_type():
    @enforce_annotations
    def target_func(x: type[SparkPayload]) -> type[SparkPayload]:
        return x
    assert target_func(SpikeArray) == SpikeArray

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_invalid_spark_type() -> None:
    @enforce_annotations
    def target_func(x: type[SparkPayload]) -> type[SparkPayload]:
        return x
    with pytest.raises(TypeError, match='Positional argument'):
        target_func(int)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_spark_config() -> None:
    @enforce_annotations
    def target_func(x: BaseSparkConfig) -> BaseSparkConfig:
        return x
    config = SparkConfig()
    assert target_func(config) == config
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################