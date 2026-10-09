spark.core.backend.transforms
=============================

.. py:module:: spark.core.backend.transforms


Attributes
----------

.. autoapisummary::

   spark.core.backend.transforms.A


Classes
-------

.. autoapisummary::

   spark.core.backend.transforms.Jit


Functions
---------

.. autoapisummary::

   spark.core.backend.transforms.jit
   spark.core.backend.transforms.scan
   spark.core.backend.transforms.eval_shape
   spark.core.backend.transforms.grad


Module Contents
---------------

.. py:data:: A

.. py:class:: Jit(fun, **options)

   A function compiled with ``jax.jit``, whose calls an open recorder records.

   `jit` creates it. Without an open recorder, a call is a call of ``jax.jit(fun, **options)``.

   While a recorder of `spark.recording` is open, each call of the function is a call of the
   recorder. Before the call, the recorder decides what to record over its steps, and `scan`
   records it along the way. After the call, the records are handed over to the recorder
   (`Recorder.push`). A call recording nothing runs the program compiled without a recorder.

   The steps of a call are those of the `scan` it runs. They are found by tracing the function
   once for each value of its static arguments. A call running no `scan` is not a call of the
   recorder.

   Called with a module among its arguments, the function is compiled with ``flax.nnx.jit``
   instead, which updates the module in place. Such calls are not recorded: recorded calls take
   the graph and the state of the model, as `spark.split` gives them.

   :param fun: The function to compile.
   :type fun: callable
   :param \*\*options: Passed to ``jax.jit``, such as ``static_argnames`` or ``donate_argnames``.

   .. attribute:: fun

      The function compiled.

      :type: callable

   .. attribute:: jitted

      ``jax.jit(fun, **options)``. Attributes not found on the `Jit`, such as ``lower`` or
      ``trace``, are read from it.

      :type: callable

   .. rubric:: Notes

   While a recorder is open:

   * The static arguments of a call give its steps, unless a `scan` of the function takes its
     length from ``xs``. The shapes of the arguments are then part of the key too.
   * The model is called within `scan` only. A call of the model outside it raises.
   * A call runs one `scan`, directly: not within another `scan`, ``jax.lax.scan``,
     ``jax.vmap``, ``jax.lax.cond`` or a function compiled apart. A `Jit` called within
     another is traced as part of it.
   * The first call of a function traces it with every probe of the recorder, without
     compiling it. A probe that cannot record the model raises there.
   * An interrupt (Ctrl-C) during a call is held until the call returns its result, and raised
     by the next call of the recorder from the loop, before it does anything. The recorder then
     counts the calls whose result the loop received. A second interrupt raises at once. A
     call is compiled before the interrupt is held.

   .. seealso::

      :py:obj:`jit`
          Creates a `Jit`.

      :py:obj:`scan`
          ``jax.lax.scan``, recorded within a `Jit` while a recorder is open.

      :py:obj:`spark.recording.Recorder`
          Decides what each call records and writes it to a run.


   .. py:attribute:: fun


   .. py:attribute:: options


   .. py:attribute:: jitted


   .. py:method:: __call__(*args, **kwargs)


   .. py:method:: warmup(*args, **kwargs)

      Compiles ahead the calls the open recorder is likely to record.

      Takes the arguments of a call. Compiles it for every probe set of
      `Recorder.warmup_sets`, with the steps of the call. Nothing is run.

      :returns: Number of probe sets compiled.
      :rtype: int

      :raises RuntimeError: When no recorder is open, or the thread opened none while several are open.
      :raises ValueError: When the call runs no `scan`.

      .. rubric:: Notes

      Warns when a set needs more memory than the device has.



   .. py:method:: __get__(instance, owner = None)


   .. py:method:: __getattr__(name)


   .. py:method:: __repr__()


.. py:function:: jit(fun = None, /, **options)

   Compiles a function with ``jax.jit``. Its calls are recorded while a recorder is open.

   :param fun: The function to compile. Without it, returns a decorator taking ``options``.
   :type fun: callable, optional
   :param \*\*options: Passed to ``jax.jit``, such as ``static_argnames`` or ``donate_argnames``.

   :returns: The compiled function. Without an open recorder, it is ``jax.jit(fun, **options)``, or
             ``flax.nnx.jit(fun, **options)`` for calls with modules among their arguments.
   :rtype: Jit

   .. seealso::

      :py:obj:`Jit`
          How a call is recorded.

      :py:obj:`scan`
          ``jax.lax.scan``, recorded within a `Jit` while a recorder is open.

   .. rubric:: Examples

   >>> @partial(spark.jit, static_argnames=['steps'])
   ... def run(graph, state, steps, **inputs):
   ...     def step(state, _):
   ...         model = spark.merge(graph, state)
   ...         outputs = model(**inputs)
   ...         return spark.split(model)[1], outputs
   ...     return spark.scan(step, state, length=steps)


.. py:function:: scan(f, init, xs = None, length = None, reverse = False, unroll = 1, _split_transpose = False)

   ``jax.lax.scan``, recorded within a call of a `Jit` while a recorder is open.

   Otherwise, it is ``jax.lax.scan``, and traces to the same program.

   Each step of the scan is one step of the model: ``f`` calls the model once. Within a
   recorded call, the probes of the call are recorded on every step, and the records are
   returned by the `Jit` to the recorder. What ``f`` returns is unchanged.

   :param f: As for ``jax.lax.scan``.
   :param init: As for ``jax.lax.scan``.
   :param xs: As for ``jax.lax.scan``.
   :param length: As for ``jax.lax.scan``.
   :param reverse: As for ``jax.lax.scan``.
   :param unroll: As for ``jax.lax.scan``.
   :param _split_transpose: As for ``jax.lax.scan``.

   :returns: As ``jax.lax.scan`` returns them.
   :rtype: carry, ys

   :raises RuntimeError: Within a recorded call, when ``f`` does not call the model once per step, or when the
       scan runs within another transformation.
   :raises ValueError: Within a recorded call, with ``reverse``.

   .. seealso::

      :py:obj:`jit`
          Compiles a function whose calls a recorder records.


.. py:function:: eval_shape(*args, **kwargs)

   Wrapper around flax.nnx.eval_shape, to simplify imports.


.. py:function:: grad(*args, **kwargs)

   Wrapper around flax.nnx.grad, to simplify imports.


