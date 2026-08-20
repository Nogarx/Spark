#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import jax.numpy as jnp
import numpy as np
import typing as tp
import spark.core.utils as utils

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@pytest.mark.parametrize(
    'value, axes, shape, contractible',
    [
        (5.0,                                             (1,),   (4, 6),    True),
        (jnp.full((4, 6), 5.0),                           (1,),   (4, 6),    True),
        (jnp.arange(4.0).reshape(4, 1),                   (1,),   (4, 6),    True),
        (jnp.broadcast_to(jnp.arange(4.0)[:, None], (4, 6)), (1,), (4, 6),   True),
        (jnp.broadcast_to(jnp.arange(6.0)[None, :], (4, 6)), (1,), (4, 6),   False),
        (jnp.arange(24.0).reshape(4, 6),                  (1,),   (4, 6),    False),
        (jnp.arange(6.0),                                 (1,),   (4, 6),    False),
        (jnp.full((4, 6), 5.0),                           (),     (4, 6),    True),
        (jnp.broadcast_to(jnp.arange(8.0)[:, None, None], (8, 3, 5)), (1, 2), (8, 3, 5), True),
        (jnp.arange(120.0).reshape(8, 3, 5),              (1, 2), (8, 3, 5), False),
    ]
)
def test_contract_axes_answers_whether_it_could_reduce(
        value: tp.Any,
        axes: tuple[int, ...],
        shape: tuple[int, ...],
        contractible: bool
    ) -> None:
    """
        Tests that an array answers as reducible along an axis exactly when it holds one value there, and
        that one that does not answers unchanged.
    """
    reduced, is_contractible = utils.contract_axes(value, axes, shape)
    assert is_contractible is contractible
    if not contractible:
        assert reduced.shape == value.shape

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.mark.parametrize(
    'shape, axes, expected',
    [
        ((4, 6),    (1,),   (4, 1)),
        ((4, 6),    (0,),   (1, 6)),
        ((8, 3, 5), (1, 2), (8, 1, 1)),
        ((4, 6),    (),     (4, 6)),
    ]
)
def test_contracted_shape(shape: tuple[int, ...], axes: tuple[int, ...], expected: tuple[int, ...]) -> None:
    """
        Tests that a contracted axis is kept holding a single element.
    """
    assert utils.contracted_shape(shape, axes) == expected

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_contract_axes_keeps_the_value_and_the_dimensions() -> None:
    """
        Tests that contracting answers with what the array holds, on a shape that still broadcasts.
    """
    value = jnp.broadcast_to(jnp.arange(4.0)[:, None], (4, 6))
    contracted, is_contractible = utils.contract_axes(value, (1,), (4, 6))
    assert is_contractible
    assert contracted.shape == utils.contracted_shape((4, 6), (1,))
    assert np.allclose(np.asarray(contracted), np.arange(4.0).reshape(4, 1))
    assert np.allclose(np.asarray(value + contracted), np.asarray(value) * 2)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_contract_axes_leaves_alone_what_is_not_an_array() -> None:
    """
        Tests that a value that is not an array answers unchanged, holding one value by itself.
    """
    assert utils.contract_axes(5.0, (1,), (4, 6)) == (5.0, True)
    assert utils.contract_axes('not an array', (1,), (4, 6)) == ('not an array', True)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
