#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import re
import string

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def normalize_name(name: str) -> str:
    """
        Converts any string into a consistent lowercase_snake_case format.
    """
    if not isinstance(name, str) or not name:
        raise TypeError('name must be a non-empty string.')
    # Insert underscores between acronyms and other words.
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    # Insert underscores between lowercase letters and uppercase letters.
    s = re.sub(r'([a-z])([A-Z])', r'\1_\2', s)
    # Replace any spaces or hyphens with a single underscore.
    s = re.sub(r'[-\s]+', '_', s)
    # Convert the whole string to lowercase.
    return s.lower()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def get_einsum_labels(num_dims: int, offset: int = 0) -> str:
    """
        Generates labels for a generalized dot product using Einstein notation.
    """
    if (offset + num_dims) > len(string.ascii_letters):
        raise ValueError(f'Requested up to {offset + num_dims} symbols but it is only possible to represent up to {len(string.ascii_letters)} '
                        f'different symbols. If this was intentional consider defining a custom label map.')
    return string.ascii_letters[offset:offset+num_dims]

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
