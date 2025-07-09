#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import re
import json
import inspect
import jax.numpy as jnp
from typing import Dict, List, Union, Any
from NodeGraphQt import NodeGraphMenu
from dataclasses import is_dataclass, asdict

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class MenuTree:
    """
        A smart wrapper around a NodeGraphMenu that allows for easy, hierarchical creation and access of submenus.
    """
    def __init__(self, graph_menu: NodeGraphMenu):
        if not isinstance(graph_menu, NodeGraphMenu):
            raise ValueError("MenuTree must be initialized with a valid NodeGraphMenu object.")
        
        # NodeGraphQt menu object.
        self.graph_menu_ref: NodeGraphMenu = graph_menu
        # Child MenuTree nodes.
        self.children: Dict[str, 'MenuTree'] = {}

    def __getitem__(self, path: Union[str, List[str]]) -> 'MenuTree':
        """
            Get / Create a submenu with the given name.
        """
        if not isinstance(path, (list, str)):
            raise TypeError(f'"path" must be of type str or List[str] but got type {type(path).__name__}')
        if isinstance(path, list):
            for subpath in path:
                if not isinstance(subpath, str):
                    raise TypeError(f'Every element of "path" must be of type str but got type {type(subpath).__name__}')

        name = path.pop(0) if isinstance(path, list) else path
        # Check if we have already created and stored this child submenu.
        if name not in self.children:
            # Create the NodeGraphMenu submenu
            graph_submenu = self.graph_menu_ref.add_menu(name)
            self.children[name] = MenuTree(graph_submenu)
        # Return the requested child node.
        if isinstance(path, list) and len(path) > 0:
            return self.children[name][path]
        else:
            return self.children[name]

    def add_command(self, name: str, func=None, shortcut=None):
        """
            Convenience wrapper to add a command to this menu level.
        """
        self.graph_menu_ref.add_command(self._to_human_readable(name), func, shortcut)


    def _to_human_readable(self, name: str) -> str:
        """
            Converts a string from various programming cases into a human-readable format.
        """
        if not isinstance(name, str) or not name:
            raise TypeError('name must be a non-empty string.')
        # Insert underscores between acronyms and other words.
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
        # Insert underscores between lowercase letters and uppercase letters.
        s = re.sub(r'([a-z])([A-Z])', r'\1_\2', s)
        # Replace any spaces or hyphens with a single underscore.
        s = re.sub(r'[-\s]+', '_', s)
        # Replace all underscores with spaces.
        return s.replace('_', ' ')
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _normalize_name(name: str) -> str:
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

def to_dict(obj: Any) -> Any:
    """
    Recursively converts dataclasses, lists, and dicts into pure dictionaries.
    """
    if is_dataclass(obj):
        # Use asdict to map dataclass.
        return asdict(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        # Recursively call this function on each item.
        return [to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        # Recursively call this function on each value.
        return {key: to_dict(value) for key, value in obj.items()}
    else:
        # For all other types (str, int, bool, etc.), return them as they are.
        return obj

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _to_human_readable(snake_str: str) -> str:
    """
    Converts snake_case or normalized_name to 'Title Case'
    """
    parts = snake_str.split('_')
    return ' '.join(word.capitalize() for word in parts if word)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _normalize_section_header(header: str) -> str:


    # Normalize for safe-r comparison
    norm_header = _normalize_name(header)
    for suffix in ['_param', '_params', '_parameter', '_parameters']:
        if norm_header.endswith(suffix):
            # Remove the suffix and convert to title case
            base = norm_header[: -len(suffix)]
            return _to_human_readable(base)
    return _to_human_readable(norm_header)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        
		# NOTE: SparkPayloads are also dataclasses. 
		# Flipping the order here leads to objects of type SparkPayload being detected as a dataclass.
        # For class types (like payload_type)
        if inspect.isclass(o):
            return o.__name__

        # For dataclasses, convert them to dicts and let the encoder handle the dict
        if is_dataclass(o):
            return asdict(o)
        
        # For JAX/Numpy dtypes
        if isinstance(o, (jnp.dtype)):
             return str(o)

        # Let the base class default method raise the TypeError
        return super().default(o)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
