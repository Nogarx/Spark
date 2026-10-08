#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import spark
from functools import partial

from cases import CASES, collect

# The tests of this module share fixtures computed once per worker: with pytest-xdist and --dist loadgroup,
# they run on one worker.
pytestmark = pytest.mark.xdist_group('probes')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

R = spark.recording
RN = spark.recording.runner
RR = spark.recording.reduce
RPC = spark.recording.probe_context
STEPS = 64
SIGNAL = spark.FloatArray(jnp.full((8,), 1.0, dtype=jnp.float16))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _brain():
    """
        The small brain of `cases`, built.
    """
    return CASES['brain']()[0]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture(scope='module')
def brain():
    return _brain()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture(scope='module')
def split_brain(brain):
    return spark.split((brain))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _reference(model, before):
    """
        What the probes of `TestProbeContexts` should read, taken directly from the model in the same step.
    """
    cache = model._cache
    return {
        'first_pool:out_spikes': cache['first_pool', 'out_spikes'].spikes,
        'first_pool.__call__:in_spikes': before['spiker'],
        'second_pool.__call__:in_spikes': jnp.concatenate([before['first_pool'], before['second_pool']]),
        'first_pool.soma.potential': model.first_pool.soma.potential.value,
        'second_pool.soma.potential': model.second_pool.soma.potential.value,
    }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@partial(jax.jit, static_argnames=('steps', 'probes', 'with_reference', 'packed'))
def _chunk(graph, state, signal, steps, probes=(), with_reference=False, packed=False):
    """
        A recorded chunk built from the parts `recorded_scan` uses, with the reference values of `TestProbeContexts`
        read in the same step.
    """
    start = RR.read_boundary(spark.merge(graph, state), probes)
    def values(state):
        model = spark.merge(graph, state)
        with RPC.ProbeContext(probes) as context:
            model(signal=signal)
        return context.values(model)
    accumulators = RR.init_accumulators(probes, jax.eval_shape(values, state))
    def step(carry, _):
        state, accumulators = carry
        model = spark.merge(graph, state)
        if with_reference:
            before = {name: model._cache[name, port].spikes for name, port in
                      (('spiker', 'spikes'), ('first_pool', 'out_spikes'), ('second_pool', 'out_spikes'))}
        with RPC.ProbeContext(probes) as context:
            out = model(signal=signal)
        accumulators, rows = collect(context, model, accumulators)
        reference = _reference(model, before) if with_reference else {}
        _, state = spark.split((model))
        return (state, accumulators), (out, rows, reference)
    (state, accumulators), (outs, rows, reference) = jax.lax.scan(step, (state, accumulators), None, length=steps)
    end = RR.read_boundary(spark.merge(graph, state), probes)
    records = RR.finalize(probes, rows, accumulators, start, end)
    return outs, state, RR.pack(probes, records) if packed else records, reference

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@partial(jax.jit, static_argnames=('steps',))
def _baseline_chunk(graph, state, signal, steps):
    """
        The same chunk, written without probes.
    """
    def step(state, _):
        model = spark.merge(graph, state)
        out = model(signal=signal)
        _, state = spark.split((model))
        return state, out
    state, outs = jax.lax.scan(step, state, None, length=steps)
    return outs, state

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _kernel(state, pool):
    return np.asarray(getattr(spark.merge(*state), pool).synapses.kernel.value)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestProbe:
    """
        Addresses, modes and reductions of a probe.
    """

    def test_a_port_address_is_parsed(self) -> None:
        probe = R.SummaryProbe('first_pool.soma:spikes')
        assert (probe.kind, probe.path, probe.name) == ('port', ('first_pool', 'soma'), 'spikes')

    def test_an_input_address_is_parsed(self) -> None:
        probe = R.RasterProbe('first_pool.__call__:in_spikes')
        assert (probe.kind, probe.path, probe.name) == ('port', ('first_pool', '__call__'), 'in_spikes')
        assert R.TraceProbe('__call__:signal').path == ('__call__',)

    def test_an_attribute_address_is_parsed(self) -> None:
        probe = R.SummaryProbe('first_pool.soma.potential')
        assert (probe.kind, probe.path, probe.name) == ('attribute', ('first_pool', 'soma'), 'potential')

    @pytest.mark.parametrize('address', ['', 'a:', ':b', 'a..b', 'a.b:c:d', 'a.__call__.b:c', 'a.__call__'])
    def test_a_malformed_address_is_refused(self, address) -> None:
        with pytest.raises(ValueError):
            R.SummaryProbe(address)

    def test_equal_probes_are_equal_and_hash_equal(self) -> None:
        a = R.TraceProbe('first_pool.soma.potential', units=range(4))
        b = R.TraceProbe('first_pool.soma.potential', units=[0, 1, 2, 3])
        assert a == b and hash(a) == hash(b)
        assert a != R.TraceProbe('first_pool.soma.potential', units=range(5))

    def test_a_pickled_probe_hashes_as_a_new_one_in_another_process(self, tmp_path) -> None:
        import pickle, subprocess, sys, textwrap
        (tmp_path / 'probe.pkl').write_bytes(pickle.dumps(R.SummaryProbe('first_pool:out_spikes', reduce=('active_fraction',))))
        script = textwrap.dedent(f'''
            import pickle, spark
            loaded = pickle.loads(open({str(tmp_path / 'probe.pkl')!r}, 'rb').read())
            print({{spark.recording.SummaryProbe('first_pool:out_spikes', reduce=('active_fraction',)): 'found'}}.get(loaded))
        ''')
        env = {**__import__('os').environ, 'PYTHONHASHSEED': '123', 'JAX_PLATFORMS': 'cpu'}
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300, env=env)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().splitlines()[-1] == 'found'

    def test_default_reductions(self) -> None:
        assert R.SummaryProbe('a.b').reduce == ('mean', 'std', 'min', 'max')
        assert R.DeltaProbe('a.b').reduce == ('norm',)

    def test_members_and_their_names_give_equal_probes(self) -> None:
        by_member = R.SummaryProbe('a:b', reduce=(R.SummaryReduction.ACTIVE_FRACTION, R.SummaryReduction.INACTIVE_UNIT_FRACTION))
        by_name = R.SummaryProbe('a:b', reduce=('active_fraction', 'inactive_unit_fraction'))
        assert by_member == by_name and hash(by_member) == hash(by_name)
        assert repr(by_member) == repr(by_name) and by_member.to_dict() == by_name.to_dict()
        # Kept by name, as plain strings.
        assert all(type(r) is str for r in by_member.reduce) and all(type(r) is str for r in R.DeltaProbe('a.b').reduce)
        assert R.SummaryProbe('a.b', reduce=R.SummaryReduction.MEAN).reduce == ('mean',)

    def test_every_mode_has_a_class(self) -> None:
        classes = (R.SummaryProbe, R.TraceProbe, R.RasterProbe, R.SnapshotProbe, R.DeltaProbe)
        assert [cls.mode for cls in classes] == list(R.ProbeMode)

    def test_probe_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            R.Probe('a.b')

    def test_probes_of_different_classes_differ(self) -> None:
        assert R.TraceProbe('a.b') != R.RasterProbe('a.b')
        assert R.TraceProbe('a.b').key == 'a.b@trace' and R.RasterProbe('a.b').key == 'a.b@raster'

    def test_a_probe_is_rebuilt_from_its_dict(self) -> None:
        probes = (
            R.SummaryProbe('a.b', reduce=('hist',), range=(0, 1), group=5),
            R.TraceProbe('a.b', units=(1, 2), stride=3),
            R.RasterProbe('a:b'),
            R.SnapshotProbe('a.b', group='episode'),
            R.DeltaProbe('a.b', reduce=('full',), group=5),
        )
        for probe in probes:
            assert R.Probe.from_dict(probe.to_dict()) == probe
            assert type(probe).from_dict(probe.to_dict()) == probe
        with pytest.raises(ValueError):
            R.TraceProbe.from_dict(probes[0].to_dict())
        with pytest.raises(ValueError):
            R.Probe.from_dict({'mode': 'nope', 'address': 'a.b'})

    @pytest.mark.parametrize('cls, kwargs', [
        (R.SnapshotProbe, dict(address='a:b')),
        (R.DeltaProbe, dict(address='a:b')),
        (R.SummaryProbe, dict(address='a.b', reduce=('median',))),
        (R.SummaryProbe, dict(address='a.b', reduce=())),
        (R.SummaryProbe, dict(address='a.b', reduce=('norm',))),
        (R.SummaryProbe, dict(address='a.b', reduce=('mean', 'mean'))),
        (R.DeltaProbe, dict(address='a.b', reduce=('mean',))),
        (R.SummaryProbe, dict(address='a.b', reduce=('hist',))),
        (R.SummaryProbe, dict(address='a.b', reduce=('hist',), range=(1, 0))),
        (R.SummaryProbe, dict(address='a.b', reduce=('hist',), range=(-np.inf, np.inf))),
        (R.SummaryProbe, dict(address='a.b', reduce=('hist',), range=(0, 1e39))),
        (R.SummaryProbe, dict(address='a.b', reduce=('hist',), range=(0, 1, 5))),
        (R.SummaryProbe, dict(address='a.b', reduce=('hist',), range=(0, 1), bins=2.5)),
        (R.SummaryProbe, dict(address='a.b', group=0)),
        (R.SnapshotProbe, dict(address='a.b', group='')),
        (R.TraceProbe, dict(address='a.b', stride=2.5)),
        (R.TraceProbe, dict(address='a.b', units=())),
        (R.RasterProbe, dict(address='a.b', stride=0)),
    ])
    def test_an_inconsistent_probe_is_refused(self, cls, kwargs) -> None:
        with pytest.raises(ValueError):
            cls(**kwargs)

    @pytest.mark.parametrize('cls, field', [
        (R.TraceProbe, dict(reduce=('mean',))),
        (R.SummaryProbe, dict(units=(0,))),
        (R.SnapshotProbe, dict(stride=2)),
        (R.RasterProbe, dict(group=5)),
    ])
    def test_a_field_of_another_class_is_refused(self, cls, field) -> None:
        with pytest.raises(TypeError):
            cls('a.b', **field)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestValidate:
    """
        Checking probes against a built model.
    """

    def test_valid_probes_pass(self, brain) -> None:
        R.validate(brain, (
            R.SummaryProbe('first_pool:out_spikes'),
            R.RasterProbe('first_pool.soma:spikes'),
            R.RasterProbe('first_pool.__call__:in_spikes'),
            R.TraceProbe('__call__:signal'),
            R.TraceProbe('first_pool.soma.potential', units=(0, 15)),
            R.DeltaProbe('first_pool.synapses.kernel'),
        ))

    @pytest.mark.parametrize('address, listed', [
        ('third_pool:out_spikes', 'first_pool'),
        ('first_pool:spikes', 'out_spikes'),
        ('first_pool.stoma:spikes', 'soma'),
        ('first_pool.soma:currents', 'spikes'),
        ('__call__:sginal', 'signal'),
        ('first_pool.__call__:signal', 'in_spikes'),
        ('first_pool.soma.potencial', 'potential'),
    ])
    def test_an_unknown_name_lists_what_is_available(self, brain, address, listed) -> None:
        cls = R.SummaryProbe if ':' in address else R.TraceProbe
        with pytest.raises(ValueError, match=listed):
            R.validate(brain, (cls(address),))

    def test_units_out_of_range_are_refused(self, brain) -> None:
        with pytest.raises(ValueError, match='16 units'):
            R.validate(brain, (R.TraceProbe('first_pool.soma.potential', units=(16,)),))

    def test_repeated_keys_are_refused(self, brain) -> None:
        with pytest.raises(ValueError, match='share the key'):
            R.validate(brain, (R.SummaryProbe('first_pool.soma.potential'), R.SummaryProbe('first_pool.soma.potential', reduce=('max',))))

    def test_an_unbuilt_model_is_refused(self) -> None:
        with pytest.raises(ValueError, match='not built'):
            R.validate(spark.nn.neurons.LIFNeuron(units=(4,), seed=1), (R.SummaryProbe('soma:spikes'),))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestPatterns:
    """
        A probe whose address is a pattern stands for a probe of every address it matches.
    """

    def test_the_addresses_are_those_get_probe_targets_lists(self, brain) -> None:
        assert set(R.probe_addresses(brain)) == {target.address for target in R.get_probe_targets(brain, {'signal': SIGNAL})}

    def test_a_pattern_gives_a_probe_of_each_match_with_its_fields(self, brain) -> None:
        probes = R.expand(brain, (R.SummaryProbe('*_pool.soma:spikes', reduce=('mean',)), R.RasterProbe('spiker:spikes')))
        assert [p.address for p in probes] == ['first_pool.soma:spikes', 'second_pool.soma:spikes', 'spiker:spikes']
        assert probes[0].reduce == probes[1].reduce == R.SummaryProbe('x:y', reduce=('mean',)).reduce

    @pytest.mark.parametrize('pattern, matched', [
        ('**.soma.potential', ['first_pool.soma.potential', 'second_pool.soma.potential']),
        ('*:*', ['spiker:spikes', 'first_pool:out_spikes', 'second_pool:out_spikes', 'integrator:signal']),
        ('*.__call__:*', ['first_pool.__call__:in_spikes', 'second_pool.__call__:in_spikes']),
    ])
    def test_what_a_pattern_matches(self, brain, pattern, matched) -> None:
        assert [p.address for p in R.expand(brain, (R.TraceProbe(pattern),))] == matched

    def test_every_match_is_validated(self, brain) -> None:
        R.validate(brain, (R.TraceProbe('*_pool.soma.potential', units=(0, 7)),))
        with pytest.raises(ValueError, match='second_pool.soma.potential'):
            R.validate(brain, (R.TraceProbe('*_pool.soma.potential', units=(12,)),))

    def test_a_pattern_matching_nothing_is_refused(self, brain) -> None:
        with pytest.raises(ValueError, match='matches nothing'):
            R.validate(brain, (R.SummaryProbe('*_pool.stoma:spikes'),))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestOff:
    """
        With no probes, recording adds nothing.
    """

    def test_the_traced_program_is_the_one_without_recording(self, split_brain) -> None:
        graph, state = split_brain
        recorded = jax.make_jaxpr(lambda g, s: RN._scan(g, s, {'signal': SIGNAL}, steps=8, outputs='all')[:2])(graph, state)
        baseline = jax.make_jaxpr(lambda g, s: _baseline_chunk.__wrapped__(g, s, SIGNAL, 8))(graph, state)
        assert str(recorded) == str(baseline)

    def test_the_last_outputs_with_inputs_per_step_trace_as_without_recording(self) -> None:
        neuron, _, per_step = CASES['alif']()
        graph, state = spark.split((neuron))
        def baseline(graph, state, xs):
            shapes = jax.eval_shape(lambda st: spark.merge(graph, st)(**jax.tree.map(lambda a: a[0], xs)), state)
            last = jax.tree.map(lambda sh: jnp.zeros(sh.shape, sh.dtype), shapes)
            def step(carry, x):
                model = spark.merge(graph, carry[0])
                out = model(**x)
                return (spark.split((model))[1], out), None
            (state, last), _ = jax.lax.scan(step, (state, last), xs)
            return last, state
        steps = len(jax.tree.leaves(per_step)[0])
        recorded = jax.make_jaxpr(lambda g, s, xs: RN._scan(g, s, None, xs, steps=steps)[:2])(graph, state, per_step)
        assert str(recorded) == str(jax.make_jaxpr(baseline)(graph, state, per_step))

    def test_the_parts_of_scan_add_nothing_without_probes(self, split_brain) -> None:
        graph, state = split_brain
        recorded = jax.make_jaxpr(lambda g, s: _chunk.__wrapped__(g, s, SIGNAL, 8)[:2])(graph, state)
        baseline = jax.make_jaxpr(lambda g, s: _baseline_chunk.__wrapped__(g, s, SIGNAL, 8))(graph, state)
        assert str(recorded) == str(baseline)

    def test_nothing_is_recorded(self, split_brain) -> None:
        graph, state = split_brain
        assert RN._scan(graph, state, {'signal': SIGNAL}, steps=4)[2] == {}

    def test_the_model_steps_as_without_recording(self, split_brain) -> None:
        graph, state = split_brain
        probes = (R.TraceProbe('first_pool.soma.potential'), R.RasterProbe('first_pool:out_spikes'))
        outs, new_state, _, _ = _chunk(graph, state, SIGNAL, 16, probes)
        base_outs, base_state = _baseline_chunk(graph, state, SIGNAL, 16)
        np.testing.assert_array_equal(np.asarray(outs['action'].value), np.asarray(base_outs['action'].value))
        np.testing.assert_array_equal(_kernel((graph, new_state), 'second_pool'), _kernel((graph, base_state), 'second_pool'))

    def test_equal_probe_sets_share_one_compilation(self, split_brain) -> None:
        graph, state = split_brain
        traces = []
        @partial(jax.jit, static_argnames=('probes',))
        def counted(graph, state, probes=()):
            traces.append(probes)
            return _chunk.__wrapped__(graph, state, SIGNAL, 2, probes)[2]
        counted(graph, state, probes=())
        counted(graph, state, probes=())
        counted(graph, state, probes=(R.SummaryProbe('first_pool:out_spikes'),))
        counted(graph, state, probes=(R.SummaryProbe('first_pool:out_spikes'),))
        assert len(traces) == 2

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture(scope='module')
def probe_context_run(split_brain):
    graph, state = split_brain
    _, _, records, reference = _chunk(graph, state, SIGNAL, STEPS, TestProbeContexts.PROBES, with_reference=True)
    return RR.complete(TestProbeContexts.PROBES, jax.device_get(records)), jax.tree.map(np.asarray, reference)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.fixture(scope='module')
def modes_run(split_brain):
    graph, state = split_brain
    _, end_state, records, _ = _chunk(graph, state, SIGNAL, STEPS, TestModes.PROBES)
    return RR.complete(TestModes.PROBES, jax.device_get(records)), (graph, state), (graph, end_state)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestProbeContexts:
    """
        Ports and attributes read during a step are the values the model produced in that step.
    """

    PROBES = (
        R.TraceProbe('first_pool:out_spikes'),
        R.TraceProbe('first_pool.soma:spikes'),
        R.TraceProbe('first_pool.__call__:in_spikes'),
        R.TraceProbe('second_pool.__call__:in_spikes'),
        R.TraceProbe('__call__:signal'),
        R.TraceProbe('first_pool.soma.potential'),
        R.TraceProbe('second_pool.soma.potential'),
    )

    def test_the_model_is_active(self, probe_context_run) -> None:
        records, _ = probe_context_run
        assert records['first_pool:out_spikes@trace'].any()

    @pytest.mark.parametrize('address', [
        'first_pool:out_spikes',
        'first_pool.__call__:in_spikes',
        'second_pool.__call__:in_spikes',
        'first_pool.soma.potential',
        'second_pool.soma.potential',
    ])
    def test_the_value_matches_the_model(self, probe_context_run, address) -> None:
        records, reference = probe_context_run
        np.testing.assert_array_equal(records[f'{address}@trace'], reference[address])

    def test_a_component_port_matches_the_port_it_feeds(self, probe_context_run) -> None:
        records, _ = probe_context_run
        np.testing.assert_array_equal(records['first_pool.soma:spikes@trace'], records['first_pool:out_spikes@trace'])

    def test_the_inputs_are_the_ones_given(self, probe_context_run) -> None:
        records, _ = probe_context_run
        expected = np.broadcast_to(np.asarray(SIGNAL.value), (STEPS, 8))
        np.testing.assert_array_equal(records['__call__:signal@trace'], expected)

    def test_a_port_the_controllers_did_not_offer_is_reported(self) -> None:
        brain = _brain()
        spikes = spark.SpikeArray(jnp.zeros((8,), dtype=jnp.uint8))
        with RPC.ProbeContext((R.TraceProbe('first_pool:out_spikes'),)) as context:
            brain.first_pool(in_spikes=spikes)
        with pytest.raises(RuntimeError, match='not produced'):
            collect(context, brain, {})

    def test_a_step_is_one_row_whatever_the_number_of_probes(self) -> None:
        brain = _brain()
        with RPC.ProbeContext(self.PROBES) as context:
            brain(signal=SIGNAL)
        _, step = collect(context, brain, {})
        assert isinstance(step, RR.StepRecords) and len(jax.tree.leaves(step)) == 1
        assert set(step.unpack()) == {probe.key for probe in self.PROBES}

    def test_probe_contexts_do_not_nest(self) -> None:
        with RPC.ProbeContext((R.SummaryProbe('a:b'),)):
            with pytest.raises(RuntimeError, match='nest'):
                with RPC.ProbeContext((R.SummaryProbe('c:d'),)):
                    pass
        assert spark.core.recording_hooks.active_probe_context() is None

    def test_probe_targets_list_ports_in_the_order_they_are_produced(self, brain) -> None:
        ports = [target.address for target in R.get_probe_targets(brain, {'signal': SIGNAL}) if target.kind == 'port']
        assert ports[:2] == ['__call__:signal', 'spiker:spikes']
        assert ports.index('first_pool.__call__:in_spikes') < ports.index('first_pool:out_spikes') < ports.index('second_pool:out_spikes')

    def test_a_neuron_can_be_the_root(self) -> None:
        neuron = spark.nn.neurons.ALIFNeuron(units=(4,), seed=3)
        spikes = spark.SpikeArray(jnp.ones((6,), dtype=jnp.uint8))
        neuron(in_spikes=spikes)
        probes = (R.TraceProbe('__call__:in_spikes'), R.TraceProbe('soma:spikes'), R.TraceProbe('soma.potential'))
        R.validate(neuron, probes)
        with RPC.ProbeContext(probes) as context:
            out = neuron(in_spikes=spikes)
        records = collect(context, neuron, {})[1].unpack()
        np.testing.assert_array_equal(np.asarray(records['__call__:in_spikes@trace']), np.ones((6,), dtype=bool))
        np.testing.assert_array_equal(np.asarray(records['soma:spikes@trace']), np.asarray(out['out_spikes'].spikes))
        np.testing.assert_array_equal(np.asarray(records['soma.potential@trace']), np.asarray(neuron.soma.potential.value))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestModes:
    """
        Reductions over a chunk agree with the same reductions computed on the full trace.
    """

    RANGE = (-80.0, 40.0)
    PROBES = (
        R.TraceProbe('first_pool.soma.potential'),
        R.SummaryProbe('first_pool.soma.potential', reduce=('mean', 'std', 'min', 'max', 'hist'), bins=16, range=RANGE),
        R.TraceProbe('first_pool:out_spikes'),
        R.SummaryProbe('first_pool:out_spikes', reduce=('active_fraction', 'active_fraction_per_unit', 'inactive_unit_fraction')),
        R.RasterProbe('first_pool:out_spikes'),
        R.DeltaProbe('first_pool.synapses.kernel', reduce=('full', 'norm', 'mean_abs')),
        R.SnapshotProbe('second_pool.synapses.kernel'),
    )

    def test_summary_of_a_value(self, modes_run) -> None:
        records, _, _ = modes_run
        trace = records['first_pool.soma.potential@trace'].astype(np.float64)
        summary = records['first_pool.soma.potential@summary']
        np.testing.assert_allclose(summary['mean'], trace.mean(), rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(summary['std'], trace.std(), rtol=1e-4, atol=1e-4)
        assert summary['min'] == trace.min() and summary['max'] == trace.max()

    def test_histogram(self, modes_run) -> None:
        records, _, _ = modes_run
        trace = records['first_pool.soma.potential@trace'].astype(np.float32)
        expected, _ = np.histogram(trace, bins=16, range=self.RANGE)
        np.testing.assert_array_equal(records['first_pool.soma.potential@summary']['hist'], expected)

    def test_summary_of_spikes(self, modes_run) -> None:
        records, _, _ = modes_run
        spikes = records['first_pool:out_spikes@trace'].astype(np.float64)
        summary = records['first_pool:out_spikes@summary']
        np.testing.assert_allclose(summary['active_fraction'], spikes.mean(), rtol=1e-6)
        np.testing.assert_allclose(summary['active_fraction_per_unit'], spikes.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(summary['inactive_unit_fraction'], (spikes.sum(axis=0) == 0).mean(), rtol=1e-6)

    def test_raster_is_the_trace_as_bool(self, modes_run) -> None:
        records, _, _ = modes_run
        raster = records['first_pool:out_spikes@raster']
        assert raster.shape == (STEPS, 16) and raster.dtype == np.bool_
        np.testing.assert_array_equal(raster, records['first_pool:out_spikes@trace'])

    def test_delta(self, modes_run) -> None:
        records, start, end = modes_run
        change = _kernel(end, 'first_pool').astype(np.float32) - _kernel(start, 'first_pool').astype(np.float32)
        delta = records['first_pool.synapses.kernel@delta']
        assert np.abs(change).max() > 0
        np.testing.assert_array_equal(delta['full'], change)
        np.testing.assert_allclose(delta['norm'], np.linalg.norm(change), rtol=1e-5)
        np.testing.assert_allclose(delta['mean_abs'], np.abs(change).mean(), rtol=1e-5)

    def test_snapshot(self, modes_run) -> None:
        records, _, end = modes_run
        np.testing.assert_array_equal(records['second_pool.synapses.kernel@snapshot'], _kernel(end, 'second_pool'))

    def test_units_and_stride(self, split_brain, modes_run) -> None:
        graph, state = split_brain
        records, _, _ = modes_run
        probes = (
            R.TraceProbe('first_pool.soma.potential', units=(1, 4, 9), stride=3),
            R.RasterProbe('first_pool:out_spikes', units=range(2, 12)),
            R.SnapshotProbe('second_pool.synapses.kernel', units=(0, 5)),
        )
        _, _, sub, _ = _chunk(graph, state, SIGNAL, STEPS, probes)
        sub = RR.complete(probes, jax.device_get(sub))
        full = records['first_pool.soma.potential@trace']
        np.testing.assert_array_equal(np.asarray(sub['first_pool.soma.potential@trace']), full[::3][:, [1, 4, 9]])
        np.testing.assert_array_equal(np.asarray(sub['first_pool:out_spikes@raster']), records['first_pool:out_spikes@trace'][:, 2:12])
        kernel = records['second_pool.synapses.kernel@snapshot'].reshape(-1)
        np.testing.assert_array_equal(np.asarray(sub['second_pool.synapses.kernel@snapshot']), kernel[[0, 5]])

    def test_units_of_a_large_value_are_selected_every_step(self, monkeypatch) -> None:
        # Past SETTINGS.select_per_step units, a chunk keeps only the units asked for on each step.
        neuron = spark.nn.neurons.LIFNeuron(units=(R.SETTINGS.select_per_step,), seed=5)
        spikes = spark.SpikeArray((jax.random.uniform(jax.random.key(0), (64,)) < 0.3).astype(jnp.uint8))
        neuron(in_spikes=spikes)
        graph, state = spark.split((neuron))
        units = R.presets.sample_units('soma', R.SETTINGS.select_per_step, 37)
        run = lambda probes: (lambda st: RN._scan(graph, st, {'in_spikes': spikes}, steps=5, probes=probes)[2])
        full = jax.jit(run((R.TraceProbe('soma.potential'), R.TraceProbe('soma:spikes'))))(state).unpack()
        sampled = (R.TraceProbe('soma.potential', units=units, stride=2), R.RasterProbe('soma:spikes', units=units))
        out = jax.jit(run(sampled))(state).unpack()
        np.testing.assert_array_equal(out['soma.potential@trace'], full['soma.potential@trace'][::2][:, list(units)])
        np.testing.assert_array_equal(out['soma:spikes@raster'], full['soma:spikes@trace'][:, list(units)] != 0)
        # The rows stacked by the scan hold the 37 potentials and the 37 spikes, as bits, of each step.
        width = 37 * np.dtype(out['soma.potential@trace'].dtype).itemsize + 5
        scan = next(e for e in jax.make_jaxpr(run(sampled))(state).jaxpr.eqns if e.primitive.name == 'scan')
        assert [v.aval.shape for v in scan.outvars if v.aval.shape[:1] == (5,)] == [(5, width)]
        # Past SETTINGS.spaced_rows_limit bytes, the potentials of the steps kept by the stride are carried in a buffer
        # of 3 rows and a spare one.
        monkeypatch.setattr(R.SETTINGS, 'spaced_rows_limit', 0)
        scan = next(e for e in jax.make_jaxpr(run(sampled))(state).jaxpr.eqns if e.primitive.name == 'scan')
        assert [v.aval.shape for v in scan.outvars if v.aval.shape[:1] == (5,)] == [(5, 5)]
        assert [v.aval.shape for v in scan.outvars if v.aval.shape == (4, 37)] == [(4, 37)]

    @pytest.mark.parametrize('steps', [10, 7, 1])
    @pytest.mark.parametrize('pack_steps', [True, False])
    @pytest.mark.parametrize('spaced', [False, True])
    def test_strides_keep_every_stride_th_step_of_the_chunk(self, split_brain, steps, pack_steps, spaced, monkeypatch) -> None:
        graph, state = split_brain
        if spaced:
            monkeypatch.setattr(R.SETTINGS, 'spaced_rows_limit', 0)             # the steps kept written to buffers of their own
        full = (R.TraceProbe('first_pool.soma.potential'), R.TraceProbe('first_pool:out_spikes'))
        strided = (
            R.TraceProbe('first_pool.soma.potential', stride=3), R.RasterProbe('first_pool:out_spikes', stride=4, units=(0, 5, 7)),
            R.TraceProbe('first_pool:out_spikes', stride=steps + 2),
        )
        run = lambda probes: jax.jit(lambda st: RN._scan(
            graph, st, {'signal': SIGNAL}, steps=steps, probes=probes, outputs='none', pack_steps=pack_steps,
        )[2])(state).unpack()
        whole, kept = run(full), run(strided)
        potential, spikes = whole['first_pool.soma.potential@trace'], whole['first_pool:out_spikes@trace']
        np.testing.assert_array_equal(kept['first_pool.soma.potential@trace'], potential[::3])
        np.testing.assert_array_equal(kept['first_pool:out_spikes@raster'], spikes[::4][:, [0, 5, 7]] != 0)
        np.testing.assert_array_equal(kept['first_pool:out_spikes@trace'], spikes[:1])

    def test_a_large_histogram_is_counted_step_by_step(self, split_brain, monkeypatch) -> None:
        # Past SETTINGS.histogram_rows_limit bytes in the chunk, nothing of the size of the value is stacked over the
        # chunk; below, the values are stacked and counted once. Both count the same.
        graph, state = split_brain
        probes = (R.SummaryProbe('first_pool.soma.potential', reduce=('hist', 'mean'), bins=8, range=(-80.0, 40.0)),)
        # A new function each time: traces are cached by function.
        chunk = lambda: (lambda st: RN._scan(graph, st, {'signal': SIGNAL}, steps=50, probes=probes, outputs='none')[2])
        stacked = lambda: [v.aval.shape for v in next(
            e for e in jax.make_jaxpr(chunk())(state).jaxpr.eqns if e.primitive.name == 'scan'
        ).outvars if v.aval.shape[:1] == (50,)]
        key = probes[0].key
        once = jax.jit(chunk())(state).unpack()[key]
        assert stacked()
        monkeypatch.setattr(R.SETTINGS, 'histogram_rows_limit', 0)
        assert stacked() == []
        by_step = jax.jit(chunk())(state).unpack()[key]
        np.testing.assert_array_equal(by_step['hist'], once['hist'])
        assert once['hist'].sum() > 0 and by_step['mean'] == once['mean']

    @pytest.mark.parametrize('compare', [True, False])
    @pytest.mark.parametrize('sort', [True, False])
    @pytest.mark.parametrize('value_range, bins', [((-2.0, 2.0), 8), ((-1.0, 1.0), 1000), ((1e6, 1e6 + 1.0), 32), ((0.0, 1.0), 1)])
    def test_histogram_edges_nan_and_infinities(self, monkeypatch, compare, sort, value_range, bins) -> None:
        # Compared with every edge, or placed from the position in the range (or among the edges, when
        # they are closer than float32 tells apart, as over (1e6, 1e6 + 1)), then scattered or sorted.
        monkeypatch.setattr(R.SETTINGS, 'histogram_compare_limit', R.SETTINGS.histogram_compare_limit if compare else 0)
        monkeypatch.setattr(R.SETTINGS, 'histogram_compare_bins', R.SETTINGS.histogram_compare_bins if compare else 0)
        monkeypatch.setattr(RR, '_counts_by_sorting', lambda size: sort)
        lo, hi = value_range
        rng = np.random.default_rng(0)
        edges = np.linspace(lo, hi, bins + 1, dtype=np.float32)
        span = hi - lo
        x = np.concatenate([
            rng.uniform(lo - 0.1 * span, hi + 0.1 * span, 5000), edges, np.nextafter(edges, np.float32(np.inf)),
            np.nextafter(edges, np.float32(-np.inf)), [np.nan, np.inf, -np.inf],
        ]).astype(np.float32)
        x = x[(np.abs(x) > 1e-37) | (x == 0)]                          # no float32 subnormals, read as zero on the processor
        upper = np.concatenate([edges[1:-1], [np.nextafter(np.float32(hi), np.float32(np.inf))]])
        expected = [np.sum((x >= a) & (x < b)) for a, b in zip(edges[:-1], upper)]
        np.testing.assert_array_equal(np.asarray(jax.jit(RR._histogram, static_argnums=(1, 2))(x, bins, value_range)), expected)

    @pytest.mark.parametrize('value_range, bins', [((-3e38, 3e38), 8), ((-2e38, 2e38), 1000), ((0.0, 3e38), 3)])
    def test_histograms_of_ranges_near_the_float32_limit(self, monkeypatch, value_range, bins) -> None:
        # The width of these ranges is no float32: the values are placed among the edges.
        monkeypatch.setattr(R.SETTINGS, 'histogram_compare_limit', 0)
        monkeypatch.setattr(R.SETTINGS, 'histogram_compare_bins', 0)
        lo, hi = value_range
        edges = np.linspace(lo, hi, bins + 1, dtype=np.float32)
        x = np.random.default_rng(0).uniform(lo, hi, 20_000).astype(np.float32)
        upper = np.concatenate([edges[1:-1], [np.nextafter(np.float32(hi), np.float32(np.inf))]])
        expected = [np.sum((x >= a) & (x < b)) for a, b in zip(edges[:-1], upper)]
        np.testing.assert_array_equal(np.asarray(jax.jit(RR._histogram, static_argnums=(1, 2))(x, bins, value_range)), expected)

    def test_units_out_of_range_fail_when_traced(self, split_brain) -> None:
        graph, state = split_brain
        with pytest.raises(ValueError, match='16 units'):
            _chunk(graph, state, SIGNAL, 2, (R.TraceProbe('first_pool:out_spikes', units=(16,)),))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestPack:
    """
        Records packed into one buffer come back unchanged.
    """

    @staticmethod
    def _assert_same(got, expected) -> None:
        assert jax.tree.structure(got) == jax.tree.structure(expected)
        for path, leaf in jax.tree_util.tree_leaves_with_path(expected):
            value = got
            for entry in path:
                value = value[entry.key]
            leaf = np.asarray(leaf)
            assert value.dtype == leaf.dtype and value.shape == leaf.shape
            np.testing.assert_array_equal(value, leaf)

    def test_round_trip(self, split_brain, modes_run) -> None:
        graph, state = split_brain
        records, _, _ = modes_run
        _, _, partials, _ = _chunk(graph, state, SIGNAL, STEPS, TestModes.PROBES)
        _, _, packed, _ = _chunk(graph, state, SIGNAL, STEPS, TestModes.PROBES, packed=True)
        assert isinstance(packed, R.Packed) and len(jax.tree.leaves(packed)) == 1
        packed, partials = jax.device_get(packed), jax.device_get(partials)
        self._assert_same(packed.unpack(completed=False), partials)
        self._assert_same(packed.unpack(), records)
        # Bool records, the rasters, take one bit per entry.
        size = lambda leaf: -(-leaf.size // 8) if leaf.dtype == np.bool_ else leaf.nbytes
        assert packed.nbytes == sum(size(np.asarray(leaf)) for leaf in jax.tree.leaves(partials))

    def test_empty_records_stay_empty(self) -> None:
        assert RR.pack((), {}) == {}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestPrecision:
    """
        Reductions do not accumulate in the dtype of the value.
    """

    @staticmethod
    def _record(probe, per_step):
        """
            Records ``per_step``, stacked along a leading step axis, with ``probe``.
        """
        first = RR.step_value(probe, jax.tree.map(lambda a: a[0], per_step))
        accumulators = RR.init_accumulators((probe,), {probe.key: first})
        def step(accumulators, value):
            values = {probe.key: RR.step_value(probe, value)}
            return RR.accumulate((probe,), accumulators, values), RR.pack_step(values)
        accumulators, rows = jax.lax.scan(step, accumulators, per_step)
        return RR.complete((probe,), jax.device_get(RR.finalize((probe,), rows, accumulators)))[probe.key]

    def _reduce(self, probe, value, steps):
        return self._record(probe, jax.tree.map(lambda a: jnp.broadcast_to(a, (steps, *a.shape)), value))

    def test_spike_counts_past_float16_precision(self) -> None:
        spikes = spark.SpikeArray(jnp.ones((8,), dtype=jnp.uint8))
        probe = R.SummaryProbe('a:b', reduce=('mean', 'active_fraction', 'active_fraction_per_unit', 'inactive_unit_fraction'))
        out = self._reduce(probe, spikes, steps=4096)
        assert out['active_fraction'] == 1.0 and out['mean'] == 1.0 and out['inactive_unit_fraction'] == 0.0
        np.testing.assert_array_equal(out['active_fraction_per_unit'], np.ones((8,), dtype=np.float32))

    def test_histogram_counts_past_float16_precision(self) -> None:
        value = spark.FloatArray(jnp.full((16,), 0.5, dtype=jnp.float16))
        probe = R.SummaryProbe('a.b', reduce=('hist', 'mean'), bins=4, range=(0.0, 1.0))
        out = self._reduce(probe, value, steps=4096)
        np.testing.assert_array_equal(out['hist'], [0, 0, 4096 * 16, 0])
        assert out['mean'] == 0.5

    def test_histogram_counts_past_int32(self) -> None:
        low, high = jnp.zeros((2,), jnp.int32), jnp.zeros((2,), jnp.int32)
        add = jax.jit(RR._add_counts)
        for _ in range(5):
            low, high = add(low, high, jnp.array([2 ** 31 - 1, 3], jnp.int32))
        probe = R.SummaryProbe('a.b', reduce=('hist',), bins=2, range=(0.0, 1.0))
        out = RR.complete((probe,), {probe.key: {'hist': np.asarray(low), 'hist_high': np.asarray(high)}})[probe.key]
        np.testing.assert_array_equal(out['hist'], [5 * (2 ** 31 - 1), 15])
        assert out['hist'].dtype == np.int64

    def test_values_without_entries_are_refused(self) -> None:
        for cls in (R.SummaryProbe, R.DeltaProbe):
            with pytest.raises(ValueError, match='no entries'):
                R.probe._check_size(0, cls('a.b'))
        with pytest.raises(ValueError, match='no entries'):
            RR.step_value(R.SummaryProbe('a.b'), jnp.zeros((0, 3)))
        assert RR.step_value(R.TraceProbe('a.b'), jnp.zeros((0, 3))).shape == (0, 3)

    @pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
    def test_rounding_to_a_narrow_type_matches_numpy(self, dtype) -> None:
        import ml_dtypes
        from spark.recording.reduce import _round_to
        rng = np.random.default_rng(0)
        x = np.concatenate([
            rng.normal(0, 3, 20_000), rng.normal(0, 1e-5, 20_000), rng.normal(0, 1e-7, 20_000),
            rng.uniform(-70_000, 70_000, 20_000),
            (np.arange(-20_000, 20_000) + 0.5) * 2.0 ** -24,        # halfway points, subnormal
            (np.arange(2048, 4096) + 0.5) * 2.0 ** -10,             # halfway points, normal
            [0.0, -0.0, np.inf, -np.inf, 65504.0, 65519.0, 65520.0],
        ]).astype(np.float32)
        narrow = np.float16 if dtype == 'float16' else ml_dtypes.bfloat16
        expected = x.astype(narrow).astype(np.float32)
        np.testing.assert_array_equal(np.asarray(jax.jit(_round_to, static_argnums=1)(x, jnp.dtype(dtype))), expected)

    def test_integer_extremes_are_exact(self) -> None:
        # 2 ** 24 + 1 has no float32; the mean, kept in float32, must not round the extremes.
        value = spark.IntegerArray(jnp.array([2 ** 24 + 1, -(2 ** 24) - 1, 3], dtype=jnp.int32))
        probe = R.SummaryProbe('a.b', reduce=('mean', 'min', 'max'))
        out = self._reduce(probe, value, steps=4)
        assert (out['min'], out['max']) == (-(2 ** 24) - 1, 2 ** 24 + 1)

    def test_extremes_of_bfloat16_values(self) -> None:
        value = spark.FloatArray(jnp.array([1.5, -2.25, 0.0078125], dtype=jnp.bfloat16))
        out = self._reduce(R.SummaryProbe('a.b'), value, steps=3)
        assert (out['min'], out['max']) == (-2.25, 1.5) and out['min'].dtype == np.float32

    def test_spread_of_a_value_far_from_zero(self) -> None:
        # Alternates between 1000.0 and 1000.5: mean 1000.25, std 0.25. Summed as raw squares (about 1e6
        # each) in float32, the variance (0.0625) is lost to rounding.
        steps = jnp.tile(jnp.array([[1000.0], [1000.5]], dtype=jnp.float16), (2048, 64))
        probe = R.SummaryProbe('a.b', reduce=('mean', 'std'))
        out = self._record(probe, steps)
        assert out['mean'] == np.float32(1000.25)
        np.testing.assert_allclose(out['std'], 0.25, rtol=1e-6)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
