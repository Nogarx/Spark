#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

"""
    What `spark.recording` plugs into.

    The core and the controllers check these to know whether something records, and `spark.recording`
    sets them when it is imported and while a recorder is open. They live here so that nothing below
    `spark.recording` imports it.
"""

from __future__ import annotations
import typing as tp

if tp.TYPE_CHECKING:
    from spark.core.backend.transforms import Jit
    from spark.recording.probe_context import ProbeContext

import contextvars

# TODO: A probe context sees the outputs of modules and the inputs of controllers only. A value a module computes
# within its call, neither returned nor held in a Variable (an effective learning rate, the norm of an update),
# cannot be probed. A module-side hook offering such a value to the open probe context, as
# `self.report(name, value)`, would cover it. To validate first: whether such values are needed, and whether 0-d values record correctly
# as traces and summaries.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

OPEN_RECORDERS: dict = {}
"""
    The open recorders of `spark.recording`, in the order they were opened, with the thread that opened each.
"""

TRACED_CALL: contextvars.ContextVar[tp.Any] = contextvars.ContextVar('spark_traced_call', default=None)
"""
    The call of a `jit` function traced for a recorder, or None.
"""

PROBE_CONTEXT: contextvars.ContextVar[ProbeContext | None] = contextvars.ContextVar('spark_probe_context', default=None)
"""
    The open `ProbeContext`, or None.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def active_probe_context() -> ProbeContext | None:
    """
        Returns the open probe context.

        Returns
        -------
        ProbeContext or None
            The open probe context, or None when none is open.
    """
    return PROBE_CONTEXT.get()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class RecordingHooks(tp.Protocol):
    """
        What `spark.recording` provides to `jit` and `scan` while a recorder is open.
    """

    def call(self, function: Jit, args: tuple, kwargs: dict) -> tp.Any:
        ...

    def scan(self, f: tp.Callable, init: tp.Any, xs: tp.Any, length: int | None, reverse: bool, unroll: int | bool, split_transpose: bool) -> tp.Any:
        ...

    def warmup(self, function: Jit, args: tuple, kwargs: dict) -> int:
        ...

_hooks: RecordingHooks | None = None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def set_recording_hooks(hooks: RecordingHooks) -> None:
    """
        Installs the hooks of `spark.recording`. Called once, when it is imported.
    """
    global _hooks
    _hooks = hooks

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def recording_hooks() -> RecordingHooks | None:
    """
        Returns the hooks of `spark.recording`, or None before it is imported.
    """
    return _hooks

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
