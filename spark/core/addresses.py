#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import fnmatch

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

ANY_DEPTH = '**'
"""
    The name of a pattern standing for any number of names.
"""

WILDCARDS = frozenset('*?[')
"""
    The characters making a name a pattern.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_pattern(address: str) -> bool:
    """
        Returns whether an address holds a wildcard.

        Parameters
        ----------
        address : str
            An address or a pattern.

        Returns
        -------
        bool
    """
    return any(character in WILDCARDS for character in address)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _split(address: str) -> tuple[tuple[str, ...], str | None]:
    """
        Splits an address into its dotted names and the port after a colon, or None.
    """
    path, colon, port = address.partition(':')
    return tuple(path.split('.')), (port if colon else None)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _name_matches(pattern: str, name: str) -> bool:
    """
        Returns whether a name of a pattern matches a name of an address.
    """
    if name.startswith('__') and not pattern.startswith('__'):
        return False
    return fnmatch.fnmatchcase(name, pattern)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _names_match(pattern: tuple[str, ...], names: tuple[str, ...]) -> bool:
    """
        Returns whether the names of a pattern match the names of an address.
    """
    if not pattern:
        return not names
    head, rest = pattern[0], pattern[1:]
    if head == ANY_DEPTH:
        # As many names as come before the first one a wildcard does not match.
        for skipped in range(len(names) + 1):
            if _names_match(rest, names[skipped:]):
                return True
            if skipped < len(names) and names[skipped].startswith('__'):
                return False
        return False
    return bool(names) and _name_matches(head, names[0]) and _names_match(rest, names[1:])

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def matches(pattern: str, address: str) -> bool:
    """
        Returns whether a pattern matches an address.

        Parameters
        ----------
        pattern : str
            A pattern, or an address, which matches itself only.
        address : str
            An address.

        Returns
        -------
        bool
            Whether every name of the pattern matches the name of the address in its place, and the
            port of the pattern the port of the address, when either has one.
    """
    pattern_names, pattern_port = _split(pattern)
    names, port = _split(address)
    if (pattern_port is None) != (port is None):
        return False
    if port is not None and not _name_matches(pattern_port, port):
        return False
    return _names_match(pattern_names, names)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def select(pattern: str, addresses: tp.Iterable[str]) -> tuple[str, ...]:
    """
        Returns the addresses a pattern matches.

        Parameters
        ----------
        pattern : str
            A pattern, or an address.
        addresses : iterable of str
            The addresses to choose from.

        Returns
        -------
        tuple of str
            The addresses matched, in the order given.
    """
    return tuple(address for address in addresses if matches(pattern, address))

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
