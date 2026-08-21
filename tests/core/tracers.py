#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import jax.numpy as jnp
import numpy as np
import typing as tp
import spark.core.utils as utils
from spark.core.tracers import Tracer, RDTracer, RFSTracer, contract_tracer_args

DTYPE = jnp.float32

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def test_tracer_decays_as_an_exponential() -> None:
    """
        Tests that a trace left alone after an impulse decays by its factor on every step.
    """
    tracer = Tracer(shape=(4,), tau=5.0, dtype=DTYPE, dt=1.0)
    tracer(jnp.ones((4,), dtype=DTYPE))
    decay = float(np.exp(-1.0 / 5.0))
    for step in range(1, 6):
        value = np.asarray(tracer(jnp.zeros((4,), dtype=DTYPE)))
        assert np.allclose(value, decay ** step, atol=1e-5)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_tracer_rests_at_its_base() -> None:
    """
        Tests that a trace with nothing coming in stays at the value it rests on.
    """
    tracer = Tracer(shape=(3,), tau=5.0, base=2.0, dtype=DTYPE, dt=1.0)
    for _ in range(20):
        value = tracer(jnp.zeros((3,), dtype=DTYPE))
    assert np.allclose(np.asarray(value), 2.0, atol=1e-5)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_tracer_reset_and_masked_reset() -> None:
    """
        Tests that resetting answers the rest value, and that a masked reset reaches only what it marks.
    """
    tracer = Tracer(shape=(4,), tau=5.0, dtype=DTYPE, dt=1.0)
    tracer(jnp.ones((4,), dtype=DTYPE))
    tracer.masked_reset(jnp.array([1.0, 0.0, 1.0, 0.0], dtype=DTYPE))
    value = np.asarray(tracer.value)
    assert np.allclose(value[[0, 2]], 0.0)
    assert np.all(value[[1, 3]] > 0.0)
    tracer.reset()
    assert np.allclose(np.asarray(tracer.value), 0.0)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_tracer_holds_a_long_tau_within_reach_of_float16() -> None:
    """
        Tests that a trace stays close to what it should be for a tau long enough that the share of a step
        that survives it no longer fits a float16.

        Holding the share a step closes rather than the one it leaves is what keeps this within reach, and
        the two are one algebraic step apart, so this stands guard over the form of the update surviving
        whatever rewrites it on the way to the device.
    """
    tau, steps, units = 200.0, 1000, 512
    generator = np.random.RandomState(0)
    spikes = generator.rand(steps, units) < 0.02
    decay = float(np.exp(-1.0 / tau))
    reference = np.zeros(units)
    for step in range(steps):
        reference = decay * reference + spikes[step]
    tracer = Tracer(shape=(units,), tau=tau, dtype=jnp.float16, dt=1.0)
    for step in range(steps):
        tracer(jnp.asarray(spikes[step], dtype=jnp.float16))
    traced = np.asarray(tracer.value, dtype=np.float64)
    assert np.max(np.abs(traced - reference)) / np.max(np.abs(reference)) < 0.015

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_rd_tracer_is_the_difference_of_its_two_traces() -> None:
    """
        Tests that the rise-decay trace answers the difference of the traces it is made of.
    """
    tracer = RDTracer(shape=(2,), tau_rise=1.0, tau_decay=5.0, dtype=DTYPE, dt=1.0)
    value = np.asarray(tracer(jnp.ones((2,), dtype=DTYPE)))
    expected = np.asarray(tracer.tracer_decay.value) - np.asarray(tracer.tracer_rise.value)
    assert np.allclose(value, expected, atol=1e-6)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_rfs_tracer_blends_its_two_traces() -> None:
    """
        Tests that the rise-fast-slow trace answers its two traces blended by alpha.
    """
    tracer = RFSTracer(
        shape=(2,), alpha=0.8, tau_rise=1.0, tau_fast_decay=5.0, tau_slow_decay=50.0, dtype=DTYPE, dt=1.0,
    )
    value = np.asarray(tracer(jnp.ones((2,), dtype=DTYPE)))
    expected = (
        0.8 * np.asarray(tracer.tracer_rise_fast.value)
        + 0.2 * np.asarray(tracer.tracer_rise_slow.value)
    )
    assert np.allclose(value, expected, atol=1e-6)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@pytest.mark.parametrize(
    'values, contracted',
    [
        ({'tau': 5.0, 'scale': 1.0, 'base': 0.0},                                            True),
        ({'tau': jnp.full((4, 6), 5.0), 'scale': 1.0, 'base': 0.0},                          True),
        ({'tau': jnp.full((4, 1), 5.0), 'scale': 1.0, 'base': 0.0},                          True),
        ({'tau': jnp.arange(1.0, 25.0).reshape(4, 6), 'scale': 1.0, 'base': 0.0},            False),
        ({'tau': 5.0, 'scale': jnp.arange(1.0, 25.0).reshape(4, 6), 'base': 0.0},            False),
    ]
)
def test_contract_tracer_args(values: dict[str, tp.Any], contracted: bool) -> None:
    """
        Tests that the arguments of a tracer are reduced only when none of them varies per element.
    """
    args, is_contracted = contract_tracer_args((1,), (4, 6), **values)
    assert is_contracted is contracted
    for name, value in args.items():
        if isinstance(value, jnp.ndarray):
            assert value.shape == ((4, 1) if contracted else values[name].shape)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_contract_tracer_args_sums_the_rest_values() -> None:
    """
        Tests that the reduced trace rests at the sum of the rest values of the traces it replaces.
    """
    args, contracted = contract_tracer_args((1,), (4, 6), tau=5.0, scale=1.0, base=0.5)
    assert contracted
    assert args['base'] == 0.5 * 6

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_reduced_tracer_answers_the_sum_of_the_traces_it_replaces() -> None:
    """
        Tests the property the reduction stands on: a tracer being linear, summing the traces of a group
        answers the same as tracing their sum.
    """
    shape, axes = (4, 6), (1,)
    args, contracted = contract_tracer_args(axes, shape, tau=5.0, scale=2.0, base=0.5)
    assert contracted
    full = Tracer(shape=shape, tau=5.0, scale=2.0, base=0.5, dtype=DTYPE, dt=1.0)
    reduced = Tracer(shape=utils.contracted_shape(shape, axes), dtype=DTYPE, dt=1.0, **args)
    generator = np.random.RandomState(0)
    for _ in range(10):
        values = jnp.asarray(generator.rand(*shape) < 0.3, dtype=DTYPE)
        summed_traces = jnp.sum(full(values), axis=axes, keepdims=True)
        traced_sum = reduced(jnp.sum(values, axis=axes, keepdims=True))
        assert np.allclose(np.asarray(summed_traces), np.asarray(traced_sum), atol=1e-3)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def test_contract_tracer_args_holds_the_group_together() -> None:
    """
        Tests that arguments describing one same tracer are reduced only when every one of them can be.
    """
    varying = jnp.arange(1.0, 25.0).reshape(4, 6)
    kept, contracted = contract_tracer_args((1,), (4, 6), tau=5.0, scale=varying, base=0.5)
    assert not contracted
    assert kept['tau'] == 5.0
    assert kept['base'] == 0.5
    assert kept['scale'].shape == (4, 6)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
