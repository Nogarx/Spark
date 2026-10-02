#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pathlib
import jax
import jax.numpy as jnp
import spark

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

STEPS = 32
TUTORIALS = pathlib.Path(__file__).resolve().parents[2] / 'tutorials'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def collect(context, model, accumulators):
    """
        Records the step just taken within ``context``, as the recorded scan records it: the running
        statistics of the summaries, and the values the step stacks, packed into one row.
    """
    R = spark.recording.reduce
    values = context.values(model)
    accumulators = R.accumulate(context.probes, accumulators, values)
    rows = {p.key: values[p.key] for p in context.probes if R.uses_rows(p, accumulators)}
    return accumulators, R.pack_step(rows)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _spikes(key, shape, rate):
    return spark.SpikeArray((jax.random.uniform(key, shape) < rate).astype(jnp.uint8))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _neuron_case(cls, **config):
    def build():
        neuron = cls(units=(12,), seed=5, **config)
        keys = jax.random.split(jax.random.key(3), STEPS + 1)
        neuron(in_spikes=_spikes(keys[0], (20,), 0.3))
        per_step = jax.tree.map(lambda *a: jnp.stack(a), *[{'in_spikes': _spikes(k, (20,), 0.3)} for k in keys[1:]])
        return neuron, {}, per_step
    return build

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _brain_case():
    pool = lambda name, units, origins: spark.ModuleSpecs(
        name = name,
        module_cls = spark.nn.neurons.ALIFNeuron,
        inputs = {'in_spikes': [spark.PortMap(origin=o, port=p) for o, p in origins]},
        config = spark.nn.neurons.ALIFNeuronConfig(_s_units=units, inhibitory_rate=0.3, synapses__kernel__scale=3000),
    )
    config = spark.nn.BrainConfig(modules_specs=[
        spark.ModuleSpecs(name='spiker', module_cls=spark.nn.interfaces.PoissonSpiker,
                          inputs={'signal': [spark.PortMap('__call__', 'signal')]}),
        pool('first_pool', (16,), [('spiker', 'spikes')]),
        pool('second_pool', (8,), [('first_pool', 'out_spikes'), ('second_pool', 'out_spikes')]),
        spark.ModuleSpecs(name='integrator', module_cls=spark.nn.interfaces.ExponentialIntegrator,
                          inputs={'spikes': [spark.PortMap('second_pool', 'out_spikes')]}, outputs={'action': 'signal'},
                          config=spark.nn.interfaces.ExponentialIntegratorConfig(num_outputs=2)),
    ], seed=7)
    brain = spark.nn.Brain(config=config)
    held = {'signal': spark.FloatArray(jnp.full((8,), 1.0, dtype=jnp.float16))}
    brain(**held)
    return brain, held, {}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _cartpole_case():
    if not spark.REGISTRY.Neurons.exists('ALIFModNeuron'):
        spark.register_neuron_from_config_file('ALIFModNeuron', str(TUTORIALS / 'alif_mod_neuron.scfg'))
    brain = spark.nn.Brain(config=spark.nn.BrainConfig.from_file(str(TUTORIALS / 'ab_brain.scfg')).with_new_seeds(seed=42))
    f16 = lambda v: spark.FloatArray(jnp.asarray(v, dtype=jnp.float16))
    held = {'signal': f16([0.3, -0.5, 0.1, 0.8]), 'drift_ex': f16([-1e-4]), 'drift_in': f16([-2.5e-5])}
    rewards = jnp.zeros((STEPS, 1), dtype=jnp.float16).at[0].set(-25.0)
    per_step = {k: spark.FloatArray(rewards) for k in ('mod_a_ex', 'mod_a_in', 'mod_b_ex', 'mod_b_in')}
    brain(**held, **jax.tree.map(lambda a: a[0], per_step))
    return brain, held, per_step

#-----------------------------------------------------------------------------------------------------------------------------------------------#

CASES = {
    'brain': _brain_case,
    'cartpole': _cartpole_case,
    'lif': _neuron_case(spark.nn.neurons.LIFNeuron),
    'alif': _neuron_case(spark.nn.neurons.ALIFNeuron),
    'adex': _neuron_case(spark.nn.neurons.AdExNeuron),
}

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
