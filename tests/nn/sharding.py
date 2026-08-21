#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import os
import sys
import json
import textwrap
import subprocess
import pytest

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

_PREAMBLE = """
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=6'
os.environ['JAX_PLATFORMS'] = 'cpu'
import json, warnings
warnings.filterwarnings('ignore')
import jax, jax.numpy as jnp, numpy as np, spark

def build_brain():
    pool = lambda name, units, origin, port: spark.ModuleSpecs(
        name=name, module_cls=spark.nn.neurons.ALIFNeuron,
        inputs={'in_spikes': [spark.PortMap(origin=origin, port=port)]},
        config=spark.nn.neurons.ALIFNeuronConfig(_s_units=units, inhibitory_rate=0.3))
    config = spark.nn.BrainConfig(modules_specs=[
        spark.ModuleSpecs(name='spiker', module_cls=spark.nn.interfaces.PoissonSpiker,
                          inputs={'signal': [spark.PortMap(origin='__call__', port='signal')]}),
        pool('first_pool', (16,), 'spiker', 'spikes'),
        pool('second_pool', (8,), 'first_pool', 'out_spikes'),
        spark.ModuleSpecs(name='integrator', module_cls=spark.nn.interfaces.ExponentialIntegrator,
                          inputs={'spikes': [spark.PortMap(origin='second_pool', port='out_spikes')]},
                          outputs={'action': 'signal'},
                          config=spark.nn.interfaces.ExponentialIntegratorConfig(num_outputs=2))])
    brain = spark.nn.Brain(config=config)
    brain(signal=spark.FloatArray(jnp.zeros((8,), dtype=jnp.float16)))
    return brain

def devices_of(array):
    while not isinstance(array, jax.Array) and hasattr(array, 'value'):
        array = array.value
    return sorted(device.id for device in array.devices())
"""

def _run(script: str) -> dict:
    """
        Runs a script in its own interpreter and reads the document it printed.
    """
    completed = subprocess.run(
        [sys.executable, '-c', _PREAMBLE + textwrap.dedent(script)],
        capture_output=True, text=True, timeout=900,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestSharding:
    """
        A model spread over more than one device (tutorial #7).
    """

    def test_six_devices_are_simulated(self) -> None:
        result = _run("print(json.dumps({'devices': len(jax.devices())}))")
        assert result['devices'] == 6

    def test_a_model_lands_on_one_device_by_default(self) -> None:
        result = _run("""
            brain = build_brain()
            print(json.dumps({
                'potential': devices_of(brain.first_pool.soma.potential),
                'kernel': devices_of(brain.first_pool.synapses.kernel),
            }))
        """)
        assert result['potential'] == [0]
        assert result['kernel'] == [0]

    def test_a_mesh_spreads_it_over_every_device(self) -> None:
        result = _run("""
            mesh = jax.sharding.Mesh(jax.devices(), axis_names=('device'),
                                     axis_types=(jax.sharding.AxisType.Auto,))
            jax.set_mesh(mesh)
            brain = build_brain()
            print(json.dumps({
                'potential': devices_of(brain.first_pool.soma.potential),
                'cache': devices_of(brain._cache['second_pool', 'out_spikes']),
            }))
        """)
        assert result['potential'] == [0, 1, 2, 3, 4, 5]
        assert result['cache'] == [0, 1, 2, 3, 4, 5]

    def test_a_module_can_be_pinned_to_a_device_of_its_own(self) -> None:
        result = _run("""
            from jax.sharding import Mesh, NamedSharding, PartitionSpec
            devices = jax.devices()
            everywhere = NamedSharding(Mesh(devices, axis_names=('device')), PartitionSpec())
            one_device = [NamedSharding(Mesh([d], axis_names=('device')), PartitionSpec()) for d in devices]
            placement = {'spiker': 0, 'integrator': 1, 'first_pool': 2, 'second_pool': 3}

            def sharding_of(path, leaf):
                head = [str(p.key) for p in path if hasattr(p, 'key')][0]
                return one_device[placement[head]] if head in placement else everywhere

            brain = build_brain()
            graph, state = spark.split((brain))
            state = jax.device_put(state, jax.tree_util.tree_map_with_path(sharding_of, state))
            brain = spark.merge(graph, state)
            print(json.dumps({
                'first_pool': devices_of(brain.first_pool.soma.potential),
                'second_pool': devices_of(brain.second_pool.soma.potential),
                'integrator': devices_of(brain.integrator.trace.tracer_rise.trace),
                'cache': devices_of(brain._cache['second_pool', 'out_spikes']),
            }))
        """)
        assert result['first_pool'] == [2]
        assert result['second_pool'] == [3]
        assert result['integrator'] == [1]
        assert result['cache'] == [0, 1, 2, 3, 4, 5]

    def test_a_sharded_model_still_runs(self) -> None:
        result = _run("""
            mesh = jax.sharding.Mesh(jax.devices(), axis_names=('device'),
                                     axis_types=(jax.sharding.AxisType.Auto,))
            jax.set_mesh(mesh)
            brain = build_brain()

            @jax.jit
            def step(graph, state, **inputs):
                model = spark.merge(graph, state)
                outputs = model(**inputs)
                _, state = spark.split((model))
                return outputs, state

            graph, state = spark.split((brain))
            signal = spark.FloatArray(jnp.array(np.full((8,), 0.5), dtype=jnp.float16))
            for _ in range(3):
                outputs, state = step(graph, state, signal=signal)
            action = np.asarray(outputs['action'].value)
            print(json.dumps({'shape': list(action.shape), 'finite': bool(np.isfinite(action).all())}))
        """)
        assert result['shape'] == [2]
        assert result['finite']

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
