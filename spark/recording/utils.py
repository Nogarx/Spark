#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import re
import signal
import numbers
import threading
import contextlib

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.\-]*$')
"""
    Pattern of a valid name of measurements, or of a run id.
"""

_EXPECTED = {None: 'an integer', 0: 'a non-negative integer', 1: 'a positive integer'}
"""
    What `integer` asks for, by its lowest value.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def whole(value: tp.Any) -> int | None:
    """
        Returns ``value`` as an int when it is a whole number, or None.

        Integers, NumPy integers included, and floats without a fractional part, such as ``1e5``,
        are whole numbers. Bools are not.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and float(value).is_integer():
        return int(value)
    return None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def integer(value: tp.Any, name: str, lowest: int | None = None, highest: int | None = None, owner: str | None = None) -> int:
    """
        Returns ``value`` as an int, or raises a ValueError naming ``name``.

        ``value`` must be a whole number (`whole`), at least ``lowest`` and at most ``highest`` when
        they are given.

        Parameters
        ----------
        value : object
            Value to check.
        name : str
            Name of the argument, for the error message.
        lowest, highest : int, optional
            Bounds of the value, both included.
        owner : str, optional
            What the argument belongs to, such as ``'Every'``, leading the error message.

        Returns
        -------
        int

        Raises
        ------
        ValueError
            When ``value`` is not a whole number, or is out of bounds.
    """
    number = whole(value)
    if number is not None and (lowest is None or number >= lowest) and (highest is None or number <= highest):
        return number
    expected = _EXPECTED.get(lowest, f'an integer of at least {lowest}')
    if highest is not None:
        expected += f' of at most {highest}'
    prefix = f'{owner}: ' if owner else ''
    raise ValueError(f'{prefix}"{name}" must be {expected}, got {value!r}.')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_name(value: tp.Any) -> bool:
    """
        Returns whether ``value`` is a string of letters, digits, ``_``, ``.`` and ``-``, starting
        with a letter or a digit.
    """
    return isinstance(value, str) and _NAME.match(value) is not None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def deliver_signal(signum: int, handler: tp.Any, frame: tp.Any = None) -> None:
    """
        Passes a received signal to ``handler`` as the process would without the recorder.
    """
    if handler is signal.default_int_handler:
        raise KeyboardInterrupt
    if callable(handler):
        handler(signum, frame)
    elif handler in (None, signal.SIG_DFL):
        signal.raise_signal(signum)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class HeldInterrupt:
    """
        A SIGINT received while `hold_interrupt` held it.

        Attributes
        ----------
        frame : frame or None
            The frame the signal interrupted, or None when none was received.
        handler : object
            The handler of SIGINT before the hold.
        armed : bool
            Whether a second SIGINT raises KeyboardInterrupt at once.
    """

    def __init__(self, handler: tp.Any, armed: bool) -> None:
        self.frame: tp.Any = None
        self.handler = handler
        self.armed = armed

    def arm(self) -> None:
        """
            Lets a second SIGINT raise KeyboardInterrupt at once, from now on.
        """
        self.armed = True

    def deliver(self) -> None:
        """
            Passes the SIGINT received, if any, to the handler before the hold (`deliver_signal`).
        """
        frame, self.frame = self.frame, None
        if frame is not None:
            deliver_signal(signal.SIGINT, self.handler, frame)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@contextlib.contextmanager
def hold_interrupt(armed: bool = True) -> tp.Generator[HeldInterrupt | None, None, None]:
    """
        Holds SIGINT while the block runs. A second SIGINT raises KeyboardInterrupt at once.

        Without ``armed``, a second SIGINT is held as well until `HeldInterrupt.arm`. The SIGINT
        held is not delivered when the block ends: `HeldInterrupt.deliver` delivers it.

        Parameters
        ----------
        armed : bool, default True
            Whether a second SIGINT raises from the start of the block.

        Returns
        -------
        context manager
            Yields the `HeldInterrupt`, or None away from the main thread and while SIGINT is
            ignored, where nothing is held.
    """
    if threading.current_thread() is not threading.main_thread():
        yield None
        return
    before = signal.getsignal(signal.SIGINT)
    if before == signal.SIG_IGN:
        yield None
        return
    held = HeldInterrupt(before, armed)

    def hold(signum: int, frame: tp.Any) -> None:
        if held.frame is not None and held.armed:
            raise KeyboardInterrupt
        if held.frame is None:
            held.frame = frame

    signal.signal(signal.SIGINT, hold)
    try:
        yield held
    finally:
        signal.signal(signal.SIGINT, before)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
