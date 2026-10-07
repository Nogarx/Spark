#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import functools
import contextlib
import pytest
import numpy as np
import jax
import flax.nnx as nnx
import spark
from cases import CASES, STEPS, collect

# The tests of this module share fixtures computed once per worker: with pytest-xdist and --dist loadgroup,
# they run on one worker.
pytestmark = pytest.mark.xdist_group('verification')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

R = spark.recording
RR = spark.recording.reduce
RPC = spark.recording.probe_context
HIST_RANGE = (-128.0, 128.0)
HIST_BINS = 32

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _probes(targets):
    """
        Every address in every mode it supports. Large ports are traced on a subset of units.
    """
    probes = []
    for target in targets:
        stride_units = tuple(range(1, target.size, 3))[:64] or (0,)
        probes.append(R.SummaryProbe(target.address, reduce=tuple(R.SummaryReduction), bins=HIST_BINS, range=HIST_RANGE))
        probes.append(R.RasterProbe(target.address))
        if target.kind == 'port':
            probes.append(R.TraceProbe(target.address, units=None if target.size <= 4096 else stride_units))
        else:
            probes.append(R.TraceProbe(target.address, units=stride_units, stride=2))
            probes.append(R.SnapshotProbe(target.address))
            probes.append(R.DeltaProbe(target.address, reduce=tuple(R.DeltaReduction)))
    return tuple(probes)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _modules(model, path=()):
    """
        ``(path, module)`` of every module called by a controller in the model.
    """
    from spark.nn.controllers.base import Controller
    yield path, model
    if isinstance(model, Controller):
        for name in model._modules_names:
            yield from _modules(getattr(model, name), (*path, name))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@contextlib.contextmanager
def _captured_calls(model, sink):
    """
        Wraps ``__call__`` of every module class in the model, storing inputs and outputs by module.
    """
    classes = {type(module) for _, module in _modules(model)}
    saved = {cls: cls.__dict__.get('__call__') for cls in classes}
    for cls in classes:
        original = cls.__call__
        @functools.wraps(original)
        def wrapper(self, *args, __original=original, **kwargs):
            out = __original(self, *args, **kwargs)
            sink[id(self)] = (kwargs, out)
            return out
        cls.__call__ = wrapper
    try:
        yield
    finally:
        for cls, call in saved.items():
            if call is None:
                del cls.__call__
            else:
                cls.__call__ = call

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _raw(value):
    if isinstance(value, spark.SpikeArray):
        return value.spikes
    if isinstance(value, (spark.SparkPayload, nnx.Variable)):
        return value.value
    return value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _reference(model, sink, targets):
    """
        Raw values of the per-step targets, read without the recording code.
    """
    by_path = {path: module for path, module in _modules(model)}
    values = {}
    for target in targets:
        if target.kind == 'port':
            path_str, port = target.address.split(':')
            path = tuple(path_str.split('.'))
            if path[-1] == '__call__':
                kwargs, _ = sink[id(by_path[path[:-1]])]
                values[target.address] = _raw(kwargs[port])
            else:
                _, outputs = sink[id(by_path[path])]
                values[target.address] = _raw(outputs[port])
        else:
            *path, name = target.address.split('.')
            node = model
            for segment in path:
                node = getattr(node, segment)
            values[target.address] = _raw(getattr(node, name))
    return values

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _run(model, held, per_step, probes, targets):
    graph, state = spark.split((model))
    sink = {}

    def chunk(graph, state, held, per_step):
        per_step = per_step or None
        start = RR.read_boundary(spark.merge(graph, state), probes)
        def values(state):
            called = spark.merge(graph, state)
            with RPC.ProbeContext(probes) as context:
                called(**held, **(jax.tree.map(lambda a: a[0], per_step) or {}))
            return context.values(called)
        accumulators = RR.init_accumulators(probes, jax.eval_shape(values, state))
        def step(carry, x):
            state, accumulators = carry
            called = spark.merge(graph, state)
            sink.clear()
            with RPC.ProbeContext(probes) as context:
                called(**held, **(x or {}))
            accumulators, rows = collect(context, called, accumulators)
            reference = _reference(called, sink, targets)
            _, state = spark.split((called))
            return (state, accumulators), (rows, reference)
        (end_state, accumulators), (rows, reference) = jax.lax.scan(step, (state, accumulators), per_step, length=STEPS)
        end = RR.read_boundary(spark.merge(graph, end_state), probes)
        return end_state, RR.pack(probes, RR.finalize(probes, rows, accumulators, start, end)), reference

    with _captured_calls(model, sink):
        compiled = jax.jit(chunk).lower(graph, state, held, per_step).compile()
    end_state, packed, reference = compiled(graph, state, held, per_step)
    records = jax.device_get(packed).unpack()
    reference = jax.tree.map(np.asarray, reference)
    start_values = _reference(spark.merge(graph, state), {}, [target for target in targets if target.kind == 'attribute'])
    end_values = _reference(spark.merge(graph, end_state), {}, [target for target in targets if target.kind == 'attribute'])
    return records, reference, jax.tree.map(np.asarray, start_values), jax.tree.map(np.asarray, end_values)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _manual_summary(values):
    """
        The summary reductions of ``values``, of shape ``(steps, ...)``, in NumPy and float64.
    """
    v = values.reshape(values.shape[0], -1)
    f = v.astype(np.float64)
    active = v != 0
    return {
        'mean': f.mean(),
        'std': f.std(),
        'min': v.min(),
        'max': v.max(),
        'active_fraction': active.mean(),
        'active_fraction_per_unit': active.mean(axis=0),
        'inactive_unit_fraction': (~active.any(axis=0)).mean(),
        'hist': np.histogram(f, bins=HIST_BINS, range=HIST_RANGE)[0],
    }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _close(got, expected, scale, what):
    tolerance = 1e-5 * max(1.0, scale)
    assert abs(float(got) - float(expected)) <= tolerance, f'{what}: {got} against {expected} (tolerance {tolerance})'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture(scope='module', params=sorted(CASES))
def case(request):
    model, held, per_step = CASES[request.param]()
    first = {**held, **jax.tree.map(lambda a: a[0], per_step)}
    targets = R.get_probe_targets(model, first)
    probes = _probes(targets)
    R.validate(model, probes)
    records, reference, start, end = _run(model, held, per_step, probes, targets)
    return request.param, targets, probes, records, reference, start, end

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestAgainstManualComputation:
    """
        Every probe of every address of the model agrees with the same metric computed by hand.
    """

    def test_every_address_was_recorded(self, case) -> None:
        name, targets, probes, records, reference, _, _ = case
        assert set(records) == {p.key for p in probes}
        assert set(reference) == {target.address for target in targets}
        assert len(targets) > 10

    def test_the_model_is_active(self, case) -> None:
        name, targets, _, _, reference, _, _ = case
        spikes = [reference[target.address] for target in targets if target.spikes and target.kind == 'port' and not target.address.startswith('__call__')]
        assert any(s.any() for s in spikes), f'{name}: no spikes in {STEPS} steps'

    def test_summaries(self, case) -> None:
        _, targets, _, records, reference, _, _ = case
        for target in targets:
            values = reference[target.address]
            got = records[f'{target.address}@summary']
            manual = _manual_summary(values)
            scale = float(np.max(np.abs(values.astype(np.float64)))) if values.size else 1.0
            _close(got['mean'], manual['mean'], scale, f'{target.address} mean')
            _close(got['std'], manual['std'], scale, f'{target.address} std')
            assert got['min'] == np.float32(manual['min']), f'{target.address} min'
            assert got['max'] == np.float32(manual['max']), f'{target.address} max'
            _close(got['active_fraction'], manual['active_fraction'], 1.0, f'{target.address} active_fraction')
            _close(got['inactive_unit_fraction'], manual['inactive_unit_fraction'], 1.0, f'{target.address} inactive_unit_fraction')
            np.testing.assert_allclose(got['active_fraction_per_unit'], manual['active_fraction_per_unit'], rtol=0, atol=1e-6, err_msg=f'{target.address} active_fraction_per_unit')
            np.testing.assert_array_equal(got['hist'], manual['hist'], err_msg=f'{target.address} hist')

    def test_traces(self, case) -> None:
        _, targets, probes, records, reference, _, _ = case
        traces = {p.address: p for p in probes if p.mode == 'trace'}
        for target in targets:
            probe, values = traces[target.address], reference[target.address]
            expected = values[::probe.stride]
            if probe.units is not None:
                expected = expected.reshape(expected.shape[0], -1)[:, list(probe.units)]
            np.testing.assert_array_equal(records[probe.key], expected, err_msg=target.address)

    def test_rasters(self, case) -> None:
        _, targets, _, records, reference, _, _ = case
        for target in targets:
            values = reference[target.address]
            expected = values.reshape(values.shape[0], -1) != 0
            np.testing.assert_array_equal(records[f'{target.address}@raster'], expected, err_msg=target.address)

    def test_snapshots(self, case) -> None:
        _, targets, _, records, _, _, end = case
        for target in targets:
            if target.kind == 'attribute':
                np.testing.assert_array_equal(records[f'{target.address}@snapshot'], end[target.address], err_msg=target.address)

    def test_deltas(self, case) -> None:
        _, targets, _, records, _, start, end = case
        for target in targets:
            if target.kind != 'attribute':
                continue
            change = end[target.address].astype(np.float32) - start[target.address].astype(np.float32)
            got = records[f'{target.address}@delta']
            np.testing.assert_array_equal(got['full'], change, err_msg=target.address)
            wide = change.astype(np.float64)
            norm, mean_abs = np.sqrt(np.sum(wide ** 2)), np.mean(np.abs(wide))
            _close(got['norm'], norm, norm, f'{target.address} norm')
            _close(got['mean_abs'], mean_abs, mean_abs, f'{target.address} mean_abs')

    def test_attributes_at_the_end_of_the_last_step_match_the_state(self, case) -> None:
        _, targets, _, _, reference, _, end = case
        for target in targets:
            if target.kind == 'attribute':
                np.testing.assert_array_equal(reference[target.address][-1], end[target.address], err_msg=target.address)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
