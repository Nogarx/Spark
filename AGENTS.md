# Spark: notes for coding agents

Spark (`spark_snn` on PyPI, `import spark`) builds spiking neural networks from modular components, on JAX
and Flax NNX. This file is a short orientation for agents using or changing the package. The notebooks in
`tutorials/` are the long form, and the docstrings are the reference.

## Install and test

- Python 3.12 or later. From a clone: `pip install -e ".[editor,test]"`. The `editor` extra adds PySide6,
  which the graph editor and the run viewer need; `test` adds pytest and pytest-xdist.
- `pytest` from the root runs `tests/`, where every `*.py` file is a test module. `--device=cpu|gpu|any`
  picks the JAX device. Without a display, Qt tests need `QT_QPA_PLATFORM=offscreen`.
- `-n <workers>` runs the tests in parallel. A worker holds about 1.7 GB, so memory sets the count, not the
  cores: `-n 8` runs the suite in about 100 s and 14 GB, against 400 s in one process. The tests of a module
  that computes a fixture once for all of them are kept on one worker by
  `pytestmark = pytest.mark.xdist_group('<module>')`.
- The version is the git tag of the release (setuptools-scm). There is no version number to edit.

## Models

- **Payloads** carry values between modules: `spark.SpikeArray` (spikes, with the sign of each unit),
  `CurrentArray` (pA), `PotentialArray` (mV), `FloatArray`, `IntegerArray`, `BooleanMask`. The array is
  `.value`, and `.spikes` for a `SpikeArray`.
- **Modules** (`spark.nn.Module`) take payloads by keyword and return a dict of payloads. A module is built
  on its first call, from the shapes it receives: call a model once with example inputs before splitting,
  jitting, recording or saving it.
- **Configurations** (`spark.nn.Config`) are frozen dataclasses. Keyword arguments of a module set the
  fields of its configuration: `ALIFNeuron(units=(16,), inhibitory_rate=0.2)`. `__` reaches a nested
  configuration (`synapses__kernel__scale=3000`), and a `_s_` prefix gives a field to every nested
  configuration (`_s_units=(16,)`).
- The configurations of components and interfaces (`spark.nn.DefaultConfig`) have `dtype`, `float16` by
  default (or `float32`), which is why the examples give float16 inputs; `dt`, the integration step, 1 ms
  by default; and `seed`. Those of controllers have `dt` and `seed`; `_s_dtype=jnp.float32` gives a whole
  neuron or brain float32. Time constants are in ms, potentials in mV and currents in pA; the docstring of
  each field gives its units.
- **Components** are the parts of a neuron: `spark.nn.somas`, `synapses`, `plasticity`, `delays`.
  **Neurons** combine them: `spark.nn.neurons.LIFNeuron`, `ALIFNeuron`, `AdExNeuron`. **Interfaces** turn
  signals into spikes and back: `spark.nn.interfaces.PoissonSpiker`, `LinearSpiker`, `ExponentialIntegrator`
  and others.
- **Controllers** run modules. A `Neuron` runs its components in order. A `Brain` wires modules together
  with `ModuleSpecs` and `PortMap`; on each step, every module of a brain reads what the others produced on
  the step before.

```python
config = spark.nn.BrainConfig(seed=7, modules_specs=[
    spark.ModuleSpecs(name='spiker', module_cls=spark.nn.interfaces.PoissonSpiker,
                      inputs={'signal': [spark.PortMap('__call__', 'signal')]}),
    spark.ModuleSpecs(name='pool', module_cls=spark.nn.neurons.ALIFNeuron,
                      inputs={'in_spikes': [spark.PortMap('spiker', 'spikes'), spark.PortMap('pool', 'out_spikes')]},
                      config=spark.nn.neurons.ALIFNeuronConfig(_s_units=(16,), synapses__kernel__scale=3000)),
    spark.ModuleSpecs(name='readout', module_cls=spark.nn.interfaces.ExponentialIntegrator,
                      inputs={'spikes': [spark.PortMap('pool', 'out_spikes')]}, outputs={'action': 'signal'},
                      config=spark.nn.interfaces.ExponentialIntegratorConfig(num_outputs=2)),
])
brain = spark.nn.Brain(config=config)
signal = spark.FloatArray(jnp.ones(8, dtype=jnp.float16))
brain(signal=signal)                                    # builds the brain
```

`PortMap('__call__', port)` is an input of the brain. `outputs={'action': 'signal'}` makes the `signal`
port of the module the `action` output of the brain. Configurations are saved with
`config.to_file('brain.scfg')` and read with `spark.nn.BrainConfig.from_file('brain.scfg')`; `.scfg` is
added when missing, and is also what the graph editor saves. `config.with_new_seeds(seed)` returns a
reseeded copy.

## What the package holds

Ready-made modules, by where they live. Each takes its parameters as keyword arguments, and its docstring
lists them with their units, its ports and its properties. The base classes (`Soma`, `Synapses`,
`Plasticity`, `Delays`, `Interface`, `Initializer`) are for writing new ones.

- **Neurons** (`spark.nn.neurons`) run delays, synapses, a soma and a plasticity rule, in that order. They
  take `in_spikes` and give `out_spikes`, and their weights learn by a `HebbianRule`.
  - `LIFNeuron`: leaky integrate-and-fire, with linear synapses and a 3 ms refractory period.
  - `ALIFNeuron`: as `LIFNeuron`, with exponential postsynaptic currents and an adaptive threshold.
  - `AdExNeuron`: adaptive exponential integrate-and-fire, with exponential postsynaptic currents.

  Other neurons are configurations of components (`spark.nn.NeuronConfig`), such as those the graph editor
  saves.
- **Somas** (`spark.nn.somas`): `LeakySoma`, `ExponentialSoma`, `IzhikevichSoma`, and their adaptive
  versions `AdaptiveLeakySoma`, `AdaptiveExponentialSoma` (AdEx) and `AdaptiveIzhikevichSoma`. An adaptive
  soma adds, each when its parameter is set: a refractory period (`cooldown`), a clamp of the potential
  after a spike (`clamp_duration`), an adaptive threshold (`threshold_delta`) and an adaptation current
  (`adaptation_delta`).
- **Synapses** (`spark.nn.synapses`): `LinearSynapses` (the weighted sum of the spikes, with no dynamics),
  `TracedSynapses` (an exponential postsynaptic current), `RDTracedSynapses` (rise and decay) and
  `RFSTracedSynapses` (two components). Their weights are the `kernel`.
- **Plasticity** (`spark.nn.plasticity`): `HebbianRule` (pair-based), `OjaRule`, `ZenkeRule` (triplet, with
  homeostatic consolidation) and `QuadrupletRule` (four terms). `ModulatedHebbianRule`, `ModulatedOjaRule`,
  `ModulatedZenkeRule` and `ModulatedQuadrupletRule` scale their update by a third factor, given on a
  `modulation` input.
- **Delays** (`spark.nn.delays`): `NDelays` (one delay per presynaptic unit) and `N2NDelays` (one per
  connection).
- **Interfaces** (`spark.nn.interfaces`):
  - signal to spikes: `PoissonSpiker` (stochastic rate code), `LinearSpiker` (deterministic rate code),
    and the place codes `TopologicalPoissonSpiker` and `TopologicalLinearSpiker`;
  - spikes to signal: `ExponentialIntegrator`;
  - between modules: `Concat`, `ConcatReshape`, `Sampler`, and the traces of a signal `SignalAccumulator`
    (which keeps its area) and `SignalAverage` (which keeps its level).
- **Initializers** (`spark.nn.initializers`) draw weights and other parameters: `ConstantInitializer`,
  `UniformInitializer`, `SparseUniformInitializer`, `NormalizedSparseUniformInitializer`.
- **Probe presets** (`spark.recording.presets`): `summary`, `activity` and `weights` build probes of every
  soma, synapse and interface of a model, and `default` builds measurements recorded by triggers. They suit
  a first look; the probes of an experiment are best picked by hand (see Recording).

## Running

The hand-written loop below is the primary interface. `spark.jit` and `spark.scan` are `jax.jit` and
`jax.lax.scan`, recorded while a recorder is open. Prefer them, or the JAX transforms, over the Flax NNX
transforms, which are slow here.

```python
graph, state = spark.split(brain)

@partial(spark.jit, static_argnames=['steps'])
def run(graph, state, steps, **inputs):
    def step(state, _):
        model = spark.merge(graph, state)
        out = model(**inputs)
        _, state = spark.split(model)
        return state, out
    state, outs = spark.scan(step, state, None, length=steps)
    return jax.tree.map(lambda x: x[-1], outs), state

out, state = run(graph, state, 100, signal=signal)     # out['action'] is the last step's output
brain = spark.merge(graph, state)                       # a model again, to inspect or save
```

Inputs that change step by step go in `xs` of `spark.scan`. `spark.recording.Runner` wraps such a loop for
quick use; it is a convenience, not the main path.

### Several devices

`spark.Partition(brain, {device: [module names], ...})` gives each device a sub-brain holding its modules,
and `spark.Partition.balanced(brain, devices)` chooses the parts. The sub-brains run one step at a time, side
by side, and the outputs that cross devices are copied between steps:

```python
partition = spark.Partition(brain, parts)
graphs, states = partition.split(brain)
received = partition.initial(states)
for _ in range(1000):
    outputs, states = partition.run(run, graphs, states, received, inputs, steps=1)
    received = partition.exchange(outputs)
brain = partition.merge(states)                         # the whole brain again
```

Modules reading or writing the properties of one another share a device. Over several machines
(`jax.distributed`), every process builds the same seeded brain and partition; recording a partition works
within one process. Tutorial #8 covers it.

## Recording

`spark.recording` records what a model does into run directories, and reads them back. A **probe** names
one value of the model and how it is recorded. **Measurements** are a named set of probes recorded
together, over the steps their trigger, or `recorder.record`, asks for. A **recorder** writes the run.

`rec.presets` probes every soma, synapse and interface of a model. It is a quick start, but heavy for large
models: pick the probes of an experiment by hand.

```python
rec = spark.recording
for target in rec.get_probe_targets(brain, {'signal': signal}):  # every address, with its kind, shape, dtype
    print(target)

rate = rec.SummaryProbe('pool.soma:spikes', reduce=('active_fraction', 'inactive_unit_fraction'))
potential = rec.SummaryProbe('pool.soma.potential', reduce=('mean', 'std', 'hist'), range=(-80.0, 40.0))
raster = rec.RasterProbe('pool:out_spikes', units=range(8))
trace = rec.TraceProbe('pool.soma.potential', units=range(8), stride=10)
change = rec.DeltaProbe('pool.synapses.kernel', reduce=('norm', 'mean_abs'))
kernel = rec.SnapshotProbe('pool.synapses.kernel')
rec.validate(brain, (rate, potential, raster, trace, change, kernel))  # raises, naming what is wrong

measurements = [
    rec.Measurements('summary', (rate, potential, change), group='episode', trigger=rec.Always()),
    rec.Measurements('activity', (raster, trace), trigger=rec.Every(5000, length=500)),
    rec.Measurements('weights', (kernel,), group='episode'),        # recorded when asked
]
with rec.Recorder('runs', brain, measurements, name='cartpole', hparams={'lr': 0.1}) as recorder:
    graph, state = spark.split(brain)
    for episode in range(100):
        recorder.tag(episode=episode)                               # a new group of 'summary' and 'weights'
        if episode % 20 == 0:
            recorder.record('weights')                              # the kernel at the end of this episode
        # Each call of a spark.jit function running a spark.scan is recorded.
        out, state = run(graph, state, 100, signal=signal)
        recorder.log(reward=reward)                                 # scalars of the host, by step
    recorder.checkpoint(spark.merge(graph, state))

saved = rec.load(recorder.path)
saved.read('summary')[3].summaries['pool.soma:spikes'].active_fraction   # episode 3
steps, rate = saved.scalar('summary/pool.soma:spikes/active_fraction')   # every episode
for start, record in saved.read('activity').items():                    # each stretch recorded
    t, spikes = record.rasters['pool:out_spikes']
```

### Rules of a recorded call

While a recorder is open, every call of a `spark.jit` function that runs a `spark.scan` is a call of the
recorder: the recorder decides what the call records before it runs, and takes the records after it. Such
a function:

- runs one `spark.scan`, directly, not within `jax.vmap`, `jax.lax.cond`, another scan or a function
  compiled apart. A `spark.jit` function called within another is traced as part of it;
- calls the model within that scan only, once per step: the steps of the call are the steps of the scan;
- runs the same number of steps for the same static arguments, or takes it from the length of `xs`;
- is traced on its first call with every probe of the recorder, without being compiled: a probe that
  cannot record the model raises there.

A `spark.jit` function that calls the model outside a `spark.scan`, such as an evaluation of one step,
raises while a recorder is open: compile it with `jax.jit`. A function that does not call the model runs as
usual. A Ctrl-C during a call is held until the call returns, and raised at the next call of the recorder.

### Probe addresses

- `path:port`: an output port of the module at `path`, read as the module produces it: `pool:out_spikes`,
  `pool.soma:spikes`.
- `path.__call__:port`: an input of the controller at `path`; `__call__:port` for an input of the model.
- `path.name`: an attribute, read after every module of the step ran: `pool.soma.potential`,
  `pool.synapses.kernel`.

`path` is the dotted chain of module names from the model. `rec.get_probe_targets` lists every address
with its shape. `rec.validate` checks probes against a built model; the sizes of the ports of nested
controllers, as `pool:out_spikes`, are checked when the recorder first traces the call.

### Probe catalogue

| Probe | Records | Parameters |
|---|---|---|
| `SummaryProbe(address, reduce=('mean', 'std', 'min', 'max'), bins=32, range=None)` | reductions over the units and the steps of each group | `reduce`: below; `bins` and `range` for `hist`, which needs `range` |
| `TraceProbe(address, units=None, stride=1)` | the value on every step, in its dtype | `units`: flat indices kept; `stride`: keeps the steps `t % stride == 0` |
| `RasterProbe(address, units=None, stride=1)` | whether each unit is nonzero on every step, stored as bits | as `TraceProbe` |
| `SnapshotProbe(address, units=None)` | an attribute after the last step of each group | attributes only |
| `DeltaProbe(address, reduce=('norm',))` | reductions of the change of an attribute over each group | `full`, `norm`, `mean_abs`; attributes only |

- Summary reductions (`rec.SummaryReduction`, or their names): `mean`, `std`, `min`, `max` over the units
  and the steps; `active_fraction`, the fraction of the units active, averaged over the steps (the firing
  rate per step, for spikes); `active_fraction_per_unit`; `inactive_unit_fraction`, the units silent over
  the whole group; `hist`.
- Reductions giving one number per group (`mean`, `std`, `min`, `max`, `active_fraction`,
  `inactive_unit_fraction`, `norm`, `mean_abs`) are also written as the scalar series
  `'<measurements>/<address>/<reduction>'`, read with `Run.scalar` and drawn by the run viewer.
- The group is set by the measurements, not by the probe: `group=1000` gives groups of 1000 steps,
  `group='episode'` one group each time the tag `episode` takes a different value. Summaries, snapshots and
  deltas need one; traces and rasters do not.

### Keeping recording light

- Summaries are reduced on the device and move one row per group. Traces and rasters move one row per step:
  keep a subset of `units` and a `stride`, and record them over a few steps (`Every(n, length=...)`,
  `recorder.record(name, steps)`, or a `When` condition).
- A snapshot copies the attribute at the end of every group: keep `units`, or groups that are rare. A
  `DeltaProbe` with `norm` or `mean_abs` follows a change without copying the weights.
- Each distinct set of measurements recorded together compiles once. Triggers whose periods divide each
  other keep the sets few, and `run.warmup(...)` (`Jit.warmup`, `Runner.warmup`) compiles them ahead.
- Calls whose length divides the size of every group never cross the end of a group, which avoids compiling
  the call that does.
- `Measurements(lookback=n)` keeps the last `n` steps on the device, written ahead of the steps recorded,
  such as the steps before a failure.

### More

- Triggers: `Manual` (the default; `recorder.record(name, steps)` asks), `Every(n, length, offset)`,
  `At(points, length)`, `Between(start, stop)`, `Always()`, `When(condition, watch, length)`. With
  `tag='episode'`, a trigger counts the values of the tag instead of steps.
- `Recorder(..., source='brain.scfg')` names the file the model was read from. Its metadata, such as the
  node positions of the graph editor, is written with the run, and the run viewer draws the graph as laid
  out in the editor; without it, the viewer lays the graph out itself.
- Host values: `recorder.log(...)` for scalars, `recorder.event(kind, ...)`, `recorder.tag(...)`, and
  `recorder.raw(name, frame)` for frames of measurements declaring `raw=(name,)`.
- Reading: `rec.load(path)`, `rec.runs(root)`, `Run.read(name)` (one `Record` per group), `Run.scalar(key)`,
  `Run.scalar_keys()`, `Run.timeline(name)`, `Run.checkpoints()`, `Run.restore(step)`.
  `Recorder.resume(path, model, step=...)` continues a run.
- Thresholds, periods and timeouts of the recorder are in `spark.recording.SETTINGS`.

## Saving and tools

- `model.checkpoint(path)` writes a `.spark` file, read back with `Class.from_checkpoint(path)`.
  `sha256=True` also writes the file's SHA-256 beside it, which `from_checkpoint(path, sha256=...)` checks.
- `spark.GraphEditor().launch()` opens the visual editor of `.scfg` models. `spark.RunViewer().open('runs')`
  opens runs: their scalars compared, and the graph of a run with its activity at each step. Both need the
  `editor` extra.

## Extending

```python
class CounterOutput(tp.TypedDict):
    total: spark.FloatArray

@spark.register_config
class CounterConfig(spark.nn.Config):
    step: float = 1.0

@spark.register_module('counter')
class Counter(spark.nn.Module):
    config: CounterConfig

    def __init__(self, config: CounterConfig | None = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.step = spark.Constant(jnp.array(self.config.step, dtype=jnp.float32))

    def build(self, signal: spark.FloatArray):                # on the first call, from the inputs given
        self.total = spark.Variable(jnp.zeros(signal.value.shape, dtype=jnp.float32))

    def reset(self) -> None:
        super().reset()
        self.total.value = jnp.zeros_like(self.total.value)

    def __call__(self, signal: spark.FloatArray) -> CounterOutput:
        self.total.value = self.total.value + self.step.value * signal.value
        return {'total': spark.FloatArray(self.total.value)}
```

- `__call__` takes its inputs by keyword and is annotated with a `TypedDict` of its outputs, from which the
  ports of the module are read.
- State that changes lives in `spark.Variable`, fixed values in `spark.Constant`.
- On the first call, the module runs `__call__` once to learn its outputs, then calls `reset`. A module whose
  state changes in `__call__` defines `reset` to return that state to its initial value; without it, that
  first run stays in the state.
- Neurons defined in editor files are registered with `spark.register_neuron_from_config_file(name, path)`
  before loading a brain that uses them.

## Repository map

- `spark/core`: modules, configurations, payloads, specs, the registry, `backend` (`jit`, `scan`, `split`,
  `merge`), checkpoints, and `recording_hooks` (what `spark.recording` plugs into).
- `spark/nn`: controllers, neurons, components, interfaces, initializers.
- `spark/recording`: probes, measurements, triggers, the recorder and its writer, `Run`, `Runner`;
  `settings.py` holds its tunables and `utils.py` its shared helpers.
- `spark/graph_editor`: the editor; `runs/` is the run viewer; `styles/` holds every style
  (`config.json` defaults, read through `styles/manager.py`).
- `tests/` mirrors `spark/`. `tutorials/` holds the notebooks, `docs/` the Sphinx site.

## Conventions for changes

- Docstrings follow the NumPy format, in a plain, neutral register: what a thing is and does, without
  justification or flourish. Code that runs while JAX traces a function says so.
- "Adaptation" names soma-side mechanisms; "modulation" is kept for synaptic plasticity.
- Files keep the separator headers between sections. Module constants go at the top, after the imports,
  or at the bottom when they need a class of the file. Widgets of the editor and the viewer take their style
  from `spark/graph_editor/styles`, not from values in the code.
- Tests of the run viewer must not write to the user's application folder: `tests/conftest.py` redirects
  the explorations, and new tests keep it so.
- The prose of the tutorials is the author's: propose changes to it rather than rewriting it.
- Commits and pull requests carry no AI attribution: no `Co-Authored-By` trailer, no "generated with" line.
