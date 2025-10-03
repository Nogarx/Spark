
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import jax
import jax.numpy as jnp
from spark import SpikeArray
from spark.nn.components.delays import N2NDelays as SynapticDelays

# TODO: These test are no longer valid, they need to be updated.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# --- Fixtures ---
# Fixtures provide a fixed baseline upon which tests can reliably and repeatedly execute.

@pytest.fixture
def test_kernel() -> jnp.ndarray:
    """
        Provides the base kernel for retrieval tests.
    """
    return jnp.array([
        [1, 3, 5, 7],
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [0, 2, 4, 6],
        [4, 4, 4, 4],
        [3, 3, 3, 3]
    ], dtype=jnp.uint8)

def test_simple_retrieval(test_kernel):
    """
        Tests delayed signaling with 1D input/output shapes.
    """
    max_delay = 12
    delays = SynapticDelays(input_shape=4, output_shape=6, max_delay=max_delay)
    delays.delay_kernel.value = test_kernel
    expected_output = jnp.zeros((6, 4), dtype=jnp.float16)

    # Phase 1: Send continuous spikes
    spikes = SpikeArray(jnp.ones((4,), dtype=jnp.float16))
    for i in range(max_delay):
        out = delays(spikes)
        rows, cols = jnp.where(delays.delay_kernel.value == i)
        expected_output = expected_output.at[rows, cols].set(1)
        assert (expected_output == out.value).all(), f'Mismatch at step {i} (sending spikes).'
    # Phase 2: Send no spikes to clear the buffer
    spikes = SpikeArray(jnp.zeros((4,), dtype=jnp.float16))
    for i in range(max_delay):
        out = delays(spikes)
        rows, cols = jnp.where(delays.delay_kernel.value == i)
        expected_output = expected_output.at[rows, cols].set(0)
        assert (expected_output == out.value).all(), f'Mismatch at step {i} (clearing buffer).'

def test_reshaped_retrieval(test_kernel):
    """
        Tests delayed signaling with multi-dimensional (2D) input/output shapes.
    """
    max_delay = 12
    delays = SynapticDelays(input_shape=(2, 2), output_shape=(3, 2), max_delay=max_delay)
    delays.delay_kernel.value = test_kernel # The object should handle internal reshaping
    reshaped_kernel = test_kernel.reshape(3, 2, 2, 2)
    expected_output = jnp.zeros((3, 2, 2, 2), dtype=jnp.float16)
    # Phase 1: Send continuous spikes
    spikes = SpikeArray(jnp.ones((2, 2), dtype=jnp.float16))
    for i in range(max_delay):
        out = delays(spikes)
        d1, d2, d3, d4 = jnp.where(reshaped_kernel == i)
        expected_output = expected_output.at[d1, d2, d3, d4].set(1)
        assert (expected_output == out.value).all(), f'Mismatch at step {i} (sending spikes).'
    # Phase 2: Send no spikes to clear the buffer
    spikes = SpikeArray(jnp.zeros((4,), dtype=jnp.float16))
    for i in range(max_delay):
        out = delays(spikes)
        d1, d2, d3, d4 = jnp.where(reshaped_kernel == i)
        expected_output = expected_output.at[d1, d2, d3, d4].set(0)
        assert (expected_output == out.value).all(), f'Mismatch at step {i} (clearing buffer).'

def test_internal_storage_buffer():
    """
        Tests the internal buffer correctly stores a history of recent spikes.
    """
    max_delay = 12
    delays = SynapticDelays(input_shape=4, output_shape=6, max_delay=max_delay)
    expected_buffer = jnp.zeros((max_delay, 4), dtype=jnp.float16)
    rng = jax.random.key(42)
    for i in range(50):
        rng, key = jax.random.split(rng, 2)
        spike_data = (jax.random.uniform(key, (4,)) < 0.5).astype(jnp.float16)
        spikes = SpikeArray(spike_data)
        # Action: process the spikes
        _ = delays(spikes)
        # Update our local copy of the expected buffer
        expected_buffer = expected_buffer.at[i % max_delay, :].set(spikes.value)
        # Assertion: check if the internal buffer matches our expectation
        internal_buffer = delays.get_dense()
        assert (expected_buffer == internal_buffer).all(), f'Buffer mismatch at step {i}.'

def test_specific_spike_pattern():
    """
        Tests the precise output timing for a specific input pattern.
    """
    delays = SynapticDelays(input_shape=2, output_shape=1, max_delay=5)
    # Input 0 has delay 1, Input 1 has delay 3
    delays.delay_kernel.value = jnp.array([[1, 3]], dtype=jnp.uint8)
    # Spike patterns to send at each time step
    spike_train = [
        jnp.array([1.0, 0.0]), # t=0
        jnp.array([0.0, 1.0]), # t=1
        jnp.array([0.0, 0.0]), # t=2
        jnp.array([1.0, 1.0]), # t=3
        jnp.array([0.0, 0.0]), # t=4
    ]
    # Expected output values for the single output neuron
    expected_outputs = [
        jnp.array([0.0, 0.0]), 
        jnp.array([1.0, 0.0]), 
        jnp.array([0.0, 0.0]), 
        jnp.array([0.0, 0.0]), 
        jnp.array([1.0, 1.0])
    ]
    for i, (spikes, expected) in enumerate(zip(spike_train, expected_outputs)):
        out = delays(SpikeArray(spikes))
        assert (out.value == expected).all(), f'Output mismatch at step {i}.'

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################