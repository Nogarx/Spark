#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import os
import sys
import json
import socket
import textwrap
import subprocess
import pytest

# The tests of this module share fixtures computed once per worker: with pytest-xdist and --dist loadgroup,
# they run on one worker.
pytestmark = pytest.mark.xdist_group('partition')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# The brain the scripts divide, and the helpers they share. Every process of a script builds the same brain from its seed.
_BRAIN = """
P = spark.PortMap

def config_of(effects=True):
    pool = lambda name, units, sources: spark.ModuleSpecs(
        name=name, module_cls=spark.nn.neurons.ALIFNeuron, inputs={'in_spikes': [P(o, p) for o, p in sources]},
        config=spark.nn.neurons.ALIFNeuronConfig(_s_units=units, synapses__kernel__scale=3000, inhibitory_rate=0.3))
    return spark.nn.BrainConfig(seed=7, modules_specs=[
        spark.ModuleSpecs(name='spiker', module_cls=spark.nn.interfaces.PoissonSpiker,
                          inputs={'signal': [P('__call__', 'signal')]}),
        pool('first_pool', (16,), [('spiker', 'spikes'), ('first_pool', 'out_spikes')]),
        pool('second_pool', (8,), [('first_pool', 'out_spikes'), ('second_pool', 'out_spikes')]),
        spark.ModuleSpecs(name='integrator', module_cls=spark.nn.interfaces.ExponentialIntegrator,
                          inputs={'spikes': [P('second_pool', 'out_spikes')]}, outputs={'action': 'signal'},
                          config=spark.nn.interfaces.ExponentialIntegratorConfig(num_outputs=2)),
        spark.ModuleSpecs(name='synapses', module_cls=spark.nn.synapses.LinearSynapses,
                          inputs={'spikes': [P('first_pool', 'out_spikes')]},
                          effects={'kernel': [P('rule', 'kernel')]} if effects else None,
                          config=spark.nn.synapses.LinearSynapsesConfig(units=(4,), kernel__scale=20000)),
        spark.ModuleSpecs(name='soma', module_cls=spark.nn.somas.LeakySoma,
                          inputs={'current': [P('synapses', 'currents')]}, outputs={'spikes': 'spikes'},
                          config=spark.nn.somas.LeakySomaConfig(units=(4,))),
        spark.ModuleSpecs(name='rule', module_cls=spark.nn.plasticity.HebbianRule,
                          inputs={'pre_spikes': [P('first_pool', 'out_spikes')], 'post_spikes': [P('soma', 'spikes')],
                                  'kernel': [P('synapses', 'kernel', is_property=True)]}),
    ])

SIGNAL = spark.FloatArray(jnp.ones((8,), dtype=jnp.float16))

def build(effects=True):
    brain = spark.nn.Brain(config=config_of(effects))
    brain(signal=SIGNAL)
    return brain

@partial(jax.jit, static_argnames=['steps'])
def run(graph, state, steps, **inputs):
    def step(state, _):
        model = spark.merge(graph, state)
        out = model(**inputs)
        _, state = spark.split(model)
        return state, out
    state, outs = jax.lax.scan(step, state, None, length=steps)
    return jax.tree.map(lambda x: x[-1], outs), state

def leaves(tree):
    return [np.asarray(leaf) for leaf in jax.tree.leaves(tree)]

def same(a, b):
    a, b = leaves(a), leaves(b)
    return len(a) == len(b) and all(x.dtype == y.dtype and np.array_equal(x, y) for x, y in zip(a, b))

def devices_of(tree):
    return sorted({device.id for leaf in jax.tree.leaves(tree) for device in leaf.devices()})

def refused(make):
    try:
        make()
    except Exception as error:
        return f'{type(error).__name__}: {error}'
    return None
"""

_SCRIPT = """
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
os.environ['JAX_PLATFORMS'] = 'cpu'
import json, tempfile, warnings
warnings.filterwarnings('ignore')
from functools import partial
import jax, jax.numpy as jnp, numpy as np, spark
""" + _BRAIN + """
STEPS = 200
d = jax.devices()
PARTS = {d[0]: ['spiker', 'first_pool'], d[1]: ['second_pool'], d[2]: ['integrator'], d[3]: ['synapses', 'soma', 'rule']}
result = {}

brain = build()
for _ in range(5):
    brain(signal=SIGNAL)
kernel = np.asarray(brain.synapses.kernel.value)
partition = spark.Partition(brain, PARTS)
result['parts'] = {str(device.id): list(names) for device, names in partition.parts.items()}
result['inputs'] = {
    str(device.id): sorted(spark.nn.Brain._get_controller_input_specs(config.modules_specs))
    for device, config in partition.configs.items()
}
result['outputs'] = {
    str(device.id): sorted(spark.nn.Brain._get_controller_output_specs(config.modules_specs))
    for device, config in partition.configs.items()
}

graph, state = spark.split(brain)
graphs, states = partition.split(brain)
received = partition.initial(states)
result['state_devices'] = {str(device.id): devices_of(states[device]) for device in partition.devices}
result['received_devices'] = {str(device.id): devices_of(received[device]) for device in partition.devices if received[device]}
result['different_steps'] = 0
result['spikes'] = 0
for _ in range(STEPS):
    reference, state = run(graph, state, 1, signal=SIGNAL)
    outputs = {}
    for device in partition.devices:
        outputs[device], states[device] = run(graphs[device], states[device], 1,
                                              **partition.inputs(device, {'signal': SIGNAL}), **received[device])
    received = partition.exchange(outputs)
    result['different_steps'] += not same(reference, partition.outputs(outputs))
    result['spikes'] += int(np.asarray(outputs[d[1]]['second_pool:out_spikes'].spikes).astype(bool).sum())
result['brain_outputs'] = sorted(partition.outputs(outputs))
result['initial_is_exchange'] = all(
    set(again) == set(received[device]) and all(same(again[name], received[device][name]) for name in again)
    for device, again in partition.initial(states).items()
)

merged = partition.merge(states)
reference = spark.merge(graph, state)
result['same_state'] = same(spark.split(merged)[1], spark.split(reference)[1])
result['merged_devices'] = devices_of(spark.split(merged)[1])
result['learned'] = not np.array_equal(np.asarray(merged.synapses.kernel.value), kernel)
with tempfile.TemporaryDirectory() as folder:
    path = merged.checkpoint(os.path.join(folder, 'brain'), verbose=False)
    restored = spark.nn.Brain.from_checkpoint(path, verbose=False)
    result['checkpoint'] = same(spark.split(restored)[1], spark.split(reference)[1])
result['merged_runs'] = same(merged(signal=SIGNAL), reference(signal=SIGNAL))
result['merged_to'] = devices_of(spark.split(partition.merge(states, device=d[2]))[1])

# The merged brain is split again and continues as the brain does.
graphs, states = partition.split(merged)
received = partition.initial(states)
graph, state = spark.split(merged)
reference, state = run(graph, state, 1, signal=SIGNAL)
outputs = {device: run(graphs[device], states[device], 1, **partition.inputs(device, {'signal': SIGNAL}), **received[device])[0]
           for device in partition.devices}
result['split_again'] = same(reference, partition.outputs(outputs))

result['refused'] = {
    'not_built': refused(lambda: spark.Partition(spark.nn.Brain(config=config_of()), PARTS)),
    'missing': refused(lambda: spark.Partition(brain, {**PARTS, d[3]: ['synapses', 'soma']})),
    'twice': refused(lambda: spark.Partition(brain, {**PARTS, d[2]: ['integrator', 'spiker']})),
    'unknown': refused(lambda: spark.Partition(brain, {**PARTS, d[2]: ['integrator', 'cerebellum']})),
    'empty': refused(lambda: spark.Partition(brain, {d[0]: [*PARTS[d[0]], 'second_pool', 'integrator', 'synapses', 'soma', 'rule'], d[1]: []})),
    'effect': refused(lambda: spark.Partition(brain, {**PARTS, d[3]: ['synapses', 'soma'], d[2]: ['integrator', 'rule']})),
    'property': refused(lambda: spark.Partition(build(effects=False), {**PARTS, d[3]: ['synapses', 'soma'], d[2]: ['integrator', 'rule']})),
    'inputs': refused(lambda: partition.inputs(d[0], {})),
    'balanced_not_built': refused(lambda: spark.Partition.balanced(spark.nn.Brain(config=config_of()), d)),
    'balanced_no_device': refused(lambda: spark.Partition.balanced(brain, [])),
}

# Partitions chosen for the devices: the brain, and one of two chains that do not meet.
def chains():
    pool = lambda name, source: spark.ModuleSpecs(
        name=name, module_cls=spark.nn.neurons.ALIFNeuron, inputs={'in_spikes': [P(source, 'spikes'), P(name, 'out_spikes')]},
        config=spark.nn.neurons.ALIFNeuronConfig(_s_units=(16,), synapses__kernel__scale=3000))
    spiker = lambda name: spark.ModuleSpecs(
        name=name, module_cls=spark.nn.interfaces.PoissonSpiker, inputs={'signal': [P('__call__', 'signal')]})
    chained = spark.nn.Brain(config=spark.nn.BrainConfig(seed=3, modules_specs=[
        spiker('spiker_a'), pool('pool_a', 'spiker_a'), spiker('spiker_b'), pool('pool_b', 'spiker_b'),
    ]))
    chained(signal=SIGNAL)
    return chained
parts_of = lambda partition: {str(device.id): list(names) for device, names in partition.parts.items()}
result['balanced'] = {count: parts_of(spark.Partition.balanced(brain, d[:count])) for count in (1, 2, 4)}
result['balanced_chains'] = parts_of(spark.Partition.balanced(chains(), d))
partition = spark.Partition.balanced(brain, d[:2])
graph, state = spark.split(brain)
graphs, states = partition.split(brain)
received = partition.initial(states)
result['balanced_different_steps'] = 0
for _ in range(20):
    reference, state = run(graph, state, 1, signal=SIGNAL)
    outputs = {}
    for device in partition.devices:
        outputs[device], states[device] = run(graphs[device], states[device], 1,
                                              **partition.inputs(device, {'signal': SIGNAL}), **received[device])
    received = partition.exchange(outputs)
    result['balanced_different_steps'] += not same(reference, partition.outputs(outputs))
print(json.dumps(result))
"""

_PROCESSES_SCRIPT = """
import os, sys
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
os.environ['JAX_PLATFORMS'] = 'cpu'
import json, warnings
warnings.filterwarnings('ignore')
from functools import partial
import jax
jax.config.update('jax_cpu_collectives_implementation', 'gloo')
jax.distributed.initialize(coordinator_address=f'localhost:{sys.argv[2]}', num_processes=2, process_id=int(sys.argv[1]))
import jax.numpy as jnp, numpy as np, spark
""" + _BRAIN + """
STEPS = 100
# Two devices in each process: the outputs of the first pool cross processes to the synapses and the rule, and those of
# the second pool to the integrator, while the second pool reads the first within the first process.
d = jax.devices()
PARTS = {d[0]: ['spiker', 'first_pool'], d[1]: ['second_pool'], d[2]: ['integrator'], d[3]: ['synapses', 'soma', 'rule']}
result = {'process': jax.process_index()}

brain = build()
for _ in range(5):
    brain(signal=SIGNAL)
graph, state = spark.split(brain)
result['brain'] = float(sum(leaf.astype(np.float64).sum() for leaf in leaves(state)))
partition = spark.Partition(brain, PARTS)
result['local_devices'] = [device.process_index for device in partition.local_devices]

graphs, states = partition.split(brain)
received = partition.initial(states)
result['received'] = {name: [] for payloads in received.values() for name in payloads}
result['received_in_place'] = all(devices_of(received[device]) == [device.id] for device in received if received[device])
result['different_steps'] = 0
for _ in range(STEPS):
    reference, state = run(graph, state, 1, signal=SIGNAL)
    outputs = {}
    for device in partition.local_devices:
        outputs[device], states[device] = run(graphs[device], states[device], 1,
                                              **partition.inputs(device, {'signal': SIGNAL}), **received[device])
    received = partition.exchange(outputs)
    produced = partition.outputs(outputs)
    result['different_steps'] += not same({name: reference[name] for name in produced}, produced)
    for payloads in received.values():
        for name, payload in payloads.items():
            result['received'][name].append(int(np.asarray(payload.spikes).sum()))
result['received'] = {name: sum(spikes) for name, spikes in result['received'].items()}
result['brain_outputs'] = sorted(produced)
result['initial_is_exchange'] = all(
    set(again) == set(received[device]) and all(same(again[name], received[device][name]) for name in again)
    for device, again in partition.initial(states).items()
)
merged = partition.merge(states)
result['same_state'] = same(spark.split(merged)[1], state)
result['merged_on_its_first_device'] = devices_of(spark.split(merged)[1]) == [partition.local_devices[0].id]
jax.distributed.shutdown()
print(json.dumps(result))
"""

_RECORDING_SCRIPT = """
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
os.environ['JAX_PLATFORMS'] = 'cpu'
import json, tempfile, warnings
warnings.filterwarnings('ignore')
from functools import partial
import jax, jax.numpy as jnp, numpy as np, spark
""" + _BRAIN + """
rec = spark.recording
STEPS = 60
d = jax.devices()
PARTS = {d[0]: ['spiker', 'first_pool'], d[1]: ['second_pool'], d[2]: ['integrator'], d[3]: ['synapses', 'soma', 'rule']}

@partial(spark.jit, static_argnames=['steps'])
def recorded(graph, state, steps, **inputs):
    def step(state, _):
        model = spark.merge(graph, state)
        out = model(**inputs)
        _, state = spark.split(model)
        return state, out
    state, outs = spark.scan(step, state, None, length=steps)
    return jax.tree.map(lambda x: x[-1], outs), state

def measurements():
    # Probes of every part, a probe of an input of the brain, and snapshots of two devices ending the same groups.
    return [
        rec.Measurements('summary', (
            rec.SummaryProbe('first_pool.soma:spikes', reduce=('active_fraction', 'mean')),
            rec.SummaryProbe('second_pool.soma.potential', reduce=('mean', 'std', 'hist'), range=(-80.0, 40.0)),
            rec.DeltaProbe('synapses.kernel', reduce=('norm', 'mean_abs')),
        ), group=20, trigger=rec.Always()),
        rec.Measurements('activity', (
            rec.RasterProbe('first_pool:out_spikes'),
            rec.TraceProbe('integrator:signal'),
            rec.TraceProbe('__call__:signal'),
            rec.TraceProbe('second_pool.soma.potential', units=range(4), stride=3),
        ), trigger=rec.Every(25, length=10)),
        rec.Measurements('weights', (
            rec.SnapshotProbe('synapses.kernel'),
            rec.SnapshotProbe('first_pool.synapses.kernel'),
        ), group=30, trigger=rec.Always()),
    ]

def record(root, partitioned):
    brain = build()
    for _ in range(5):
        brain(signal=SIGNAL)
    with rec.Recorder(root, brain, measurements(), name='partitioned' if partitioned else 'brain') as recorder:
        if partitioned:
            partition = spark.Partition(brain, PARTS)
            graphs, states = partition.split(brain)
            received = partition.initial(states)
            for _ in range(STEPS):
                outputs, states = partition.run(recorded, graphs, states, received, {'signal': SIGNAL}, steps=1)
                received = partition.exchange(outputs)
        else:
            graph, state = spark.split(brain)
            for _ in range(STEPS):
                _, state = recorded(graph, state, steps=1, signal=SIGNAL)
    return recorder.path

def contents(run, name):
    found = {}
    for key, record in run.read(name).items():
        entry = {'t0': np.array(record.t0), 't': record.t}
        for kind in ('traces', 'rasters'):
            for address, rows in getattr(record, kind).items():
                entry[f'{kind}:{address}:t'], entry[f'{kind}:{address}'] = rows.t, rows.values
        for kind in ('summaries', 'deltas'):
            for address, reductions in getattr(record, kind).items():
                for reduction, value in reductions.items():
                    entry[f'{kind}:{address}:{reduction}'] = value
        for address, value in record.snapshots.items():
            entry[f'snapshots:{address}'] = value
        found[str(key)] = entry
    return found

def equal(a, b):
    return set(a) == set(b) and all(
        set(a[key]) == set(b[key]) and all(np.array_equal(np.asarray(a[key][k]), np.asarray(b[key][k])) for k in a[key])
        for key in a
    )

result = {}
with tempfile.TemporaryDirectory() as root:
    brain_run, partitioned_run = rec.load(record(root, False)), rec.load(record(root, True))
    result['steps'] = [brain_run.step, partitioned_run.step]
    for name in ('summary', 'activity', 'weights'):
        mine, theirs = contents(partitioned_run, name), contents(brain_run, name)
        result[name] = {'records': len(theirs), 'entries': sum(len(entry) for entry in theirs.values()), 'equal': equal(mine, theirs)}
    keys = brain_run.scalar_keys()
    result['scalars'] = {
        'keys': len(keys),
        'equal': keys == partitioned_run.scalar_keys() and all(
            all(np.array_equal(x, y) for x, y in zip(brain_run.scalar(key), partitioned_run.scalar(key))) for key in keys
        ),
    }
print(json.dumps(result))
"""

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@pytest.fixture(scope='module')
def result() -> dict:
    """
        Runs the script in its own interpreter, as the number of simulated devices is fixed when JAX starts, and
        reads the document it printed.
    """
    completed = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(_SCRIPT)],
        capture_output=True, text=True, timeout=900,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])

@pytest.fixture(scope='module')
def recording() -> dict:
    """
        Runs the script recording a brain and its partition in its own interpreter, and reads the document it printed.
    """
    completed = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(_RECORDING_SCRIPT)],
        capture_output=True, text=True, timeout=900,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])

@pytest.fixture(scope='module')
def processes() -> list[dict]:
    """
        Runs the script of several processes in two interpreters joined by ``jax.distributed``, each with two simulated
        devices, and reads the documents they printed, in the order of the processes.
    """
    with socket.socket() as probe:
        probe.bind(('localhost', 0))
        port = probe.getsockname()[1]
    launched = [
        subprocess.Popen(
            [sys.executable, '-c', textwrap.dedent(_PROCESSES_SCRIPT), str(index), str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
        for index in range(2)
    ]
    try:
        finished = [process.communicate(timeout=900) for process in launched]
    finally:
        # A process left waiting for the other, which failed, is stopped.
        for process in launched:
            if process.poll() is None:
                process.kill()
    for process, (_, errors) in zip(launched, finished):
        assert process.returncode == 0, errors[-2000:]
    return [json.loads(out.strip().splitlines()[-1]) for out, _ in finished]

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestPartition:
    """
        A brain divided among four simulated devices by its modules.
    """

    def test_the_parts_keep_the_order_of_the_brain(self, result) -> None:
        assert result['parts'] == {
            '0': ['spiker', 'first_pool'], '1': ['second_pool'], '2': ['integrator'], '3': ['synapses', 'soma', 'rule'],
        }

    def test_what_crosses_devices_is_an_input_of_the_readers_and_an_output_of_the_producer(self, result) -> None:
        assert result['inputs'] == {
            '0': ['signal'], '1': ['first_pool:out_spikes'], '2': ['second_pool:out_spikes'], '3': ['first_pool:out_spikes'],
        }
        assert result['outputs'] == {
            '0': ['first_pool:out_spikes'], '1': ['second_pool:out_spikes'], '2': ['action'], '3': ['spikes'],
        }

    def test_each_sub_brain_is_on_its_device(self, result) -> None:
        assert result['state_devices'] == {'0': [0], '1': [1], '2': [2], '3': [3]}
        assert result['received_devices'] == {'1': [1], '2': [2], '3': [3]}

    def test_every_step_gives_the_outputs_of_the_brain(self, result) -> None:
        assert result['spikes'] > 0
        assert result['brain_outputs'] == ['action', 'spikes']
        assert result['different_steps'] == 0

    def test_merge_gives_the_brain_that_ran_unpartitioned(self, result) -> None:
        assert result['learned']
        assert result['same_state']
        assert result['merged_devices'] == [0]
        assert result['merged_to'] == [2]
        assert result['merged_runs']

    def test_a_checkpoint_of_the_merged_brain_reads_back(self, result) -> None:
        assert result['checkpoint']

    def test_the_merged_brain_can_be_split_again(self, result) -> None:
        assert result['split_again']

    def test_the_initial_reads_are_those_an_exchange_gives(self, result) -> None:
        assert result['initial_is_exchange']

    @pytest.mark.parametrize('case, message', [
        ('not_built', 'RuntimeError: The brain is not built'),
        ('missing', "ValueError: Every module of the brain runs on a device: ['rule'] are in no part"),
        ('twice', 'ValueError: "spiker" is in the parts of devices'),
        ('unknown', 'ValueError: "cerebellum" is not a module of the brain'),
        ('empty', 'ValueError: The part of device'),
        ('effect', 'ValueError: The property "kernel" of module "synapses" is written by module "rule"'),
        ('property', 'ValueError: Module "rule" reads the property "kernel" of module "synapses"'),
        ('inputs', "KeyError: \"Missing inputs of the brain: ['signal'].\""),
        ('balanced_not_built', 'RuntimeError: The brain is not built'),
        ('balanced_no_device', 'ValueError: A brain is divided among one device or more, got none.'),
    ])
    def test_what_cannot_be_divided_is_refused(self, result, case, message) -> None:
        assert result['refused'][case] is not None and result['refused'][case].startswith(message), result['refused'][case]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestBalanced:
    """
        Partitions chosen for the devices, from the work of each module and what crosses devices.
    """

    def test_one_device_takes_the_whole_brain(self, result) -> None:
        assert result['balanced']['1'] == {'0': ['spiker', 'first_pool', 'second_pool', 'integrator', 'synapses', 'soma', 'rule']}

    def test_the_heaviest_pool_is_cut_from_the_rest_where_one_output_crosses(self, result) -> None:
        assert result['balanced']['2'] == {
            '0': ['spiker', 'first_pool'], '1': ['second_pool', 'integrator', 'synapses', 'soma', 'rule'],
        }

    def test_devices_that_would_only_add_crossings_get_no_part(self, result) -> None:
        assert result['balanced']['4'] == result['balanced']['2']

    def test_modules_linked_by_a_property_or_an_effect_stay_together(self, result) -> None:
        for parts in result['balanced'].values():
            assert any({'synapses', 'soma', 'rule'} <= set(part) for part in parts.values())

    def test_chains_that_do_not_meet_take_a_device_each(self, result) -> None:
        assert result['balanced_chains'] == {'0': ['spiker_a', 'pool_a'], '1': ['spiker_b', 'pool_b']}

    def test_a_balanced_partition_gives_the_outputs_of_the_brain(self, result) -> None:
        assert result['balanced_different_steps'] == 0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestRecording:
    """
        A brain divided among four simulated devices recorded as the brain is.
    """

    def test_the_run_counts_the_steps_of_the_brain(self, recording) -> None:
        assert recording['steps'] == [60, 60]

    @pytest.mark.parametrize('name', ['summary', 'activity', 'weights'])
    def test_the_records_are_those_of_the_brain(self, recording, name) -> None:
        assert recording[name]['records'] > 0 and recording[name]['entries'] > 0
        assert recording[name]['equal']

    def test_the_scalars_are_those_of_the_brain(self, recording) -> None:
        assert recording['scalars']['keys'] > 0
        assert recording['scalars']['equal']

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestProcesses:
    """
        A brain divided among two processes with two simulated devices each, exchanging every step.
    """

    def test_every_process_builds_the_same_brain(self, processes) -> None:
        assert processes[0]['brain'] == processes[1]['brain']

    def test_each_process_runs_the_parts_of_its_devices(self, processes) -> None:
        assert [result['local_devices'] for result in processes] == [[0, 0], [1, 1]]

    def test_what_crosses_processes_is_gathered(self, processes) -> None:
        assert set(processes[0]['received']) == {'first_pool:out_spikes'}
        assert set(processes[1]['received']) == {'first_pool:out_spikes', 'second_pool:out_spikes'}
        assert all(spikes > 0 for result in processes for spikes in result['received'].values())
        assert all(result['received_in_place'] for result in processes)

    def test_every_step_gives_the_outputs_of_the_brain_produced_in_each_process(self, processes) -> None:
        assert [result['brain_outputs'] for result in processes] == [[], ['action', 'spikes']]
        assert [result['different_steps'] for result in processes] == [0, 0]

    def test_the_initial_reads_are_those_an_exchange_gives(self, processes) -> None:
        assert all(result['initial_is_exchange'] for result in processes)

    def test_merge_gives_every_process_the_brain_that_ran_unpartitioned(self, processes) -> None:
        assert all(result['same_state'] for result in processes)
        assert all(result['merged_on_its_first_device'] for result in processes)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
