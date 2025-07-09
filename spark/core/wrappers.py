#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import inspect
import functools
from functools import wraps
from collections.abc import Iterable

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# TODO: Hooks could drastically simplify some of the internal logic of the modules, e.g:
# 1) using conduction delays could be factor out from the pipeline since its logic is trivial
# 2) post methods can be used for targeted record of the models
# however we need to find a way to integrate them  gracefully with jax.jit, currently they throw:
# UnexpectedTracerError: Encountered an unexpected tracer. A function transformed by JAX had a side effect
# NOTE: Alternatively, this could be integrated when building the transpiler from graphs to a flat nnx model.

class HookingMeta(type):
    """
        A metaclass that inspects a `_hooks` dictionary on a class
        and applies pre/post hooks to the specified methods.
    """
    def __new__(cls, name, bases, dct):
        # Initialize object
        new_object = super().__new__(cls, name, bases, dct)

        # Hook merge logic
        def _merge_hooks(parent_hooks: dict, child_hooks: dict) -> dict:
            merged = parent_hooks.copy()
            for method_name, hooks in child_hooks.items():
                if method_name in merged and isinstance(hooks, dict):
                    merged[method_name].update(hooks)
                else:
                    merged[method_name] = hooks
            return merged

        # Gather hooks
        hooks_config = {}
        for base in reversed(new_object.__mro__):
            if '_hooks' in base.__dict__:
                base_hooks = base.__dict__['_hooks']
                hooks_config = _merge_hooks(hooks_config, base_hooks)

        # Add hooks.
        for method_name, hooks in hooks_config.items():
            # Get original method
            original_method = dct.get(method_name)
            if not callable(original_method):
                # Check if call is in parent object.
                original_method = super(type(new_object),new_object).__getattribute__(method_name)
                if not callable(original_method):
                    # Validate hook
                    if not inspect.isabstract(new_object):
                        raise ValueError(f'Expected type "callable" for hook "{method_name}", got {type(method_name)}.')

            # Get the hook functions from the configuration
            pre_hook = hooks.get('pre')
            post_hook = hooks.get('post')
            # Create a wrapped method with the hooks
            wrapped_version = create_wrapper(original_method, pre_hook, post_hook, 
                                             replace_inputs=hooks.get('replace_inputs'), 
                                             replace_outputs=hooks.get('replace_outputs'))
            # Replace the original method with the wrapped version
            setattr(new_object, method_name, wrapped_version)

        return new_object
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

def create_wrapper(original_method, pre_hook=None, post_hook=None, replace_inputs=False, replace_outputs=False):
    """
        Wrapper factory.
    """

    if not ((type(replace_inputs) == bool) or (replace_inputs == None)):
        raise ValueError(f'"replace_inputs" must be of type bool or None, got {replace_inputs}.')
    if not ((type(replace_outputs) == bool) or (replace_outputs == None)):
        raise ValueError(f'"replace_outputs" must be of type bool or None, got {replace_outputs}.')

    @wraps(original_method)
    def wrapped_method(self, *args, **kwargs):
        # Pre-hook
        if pre_hook:
            if callable(pre_hook):
                pre_hook(self, *args, **kwargs)
            elif isinstance(pre_hook, str):
                internal_method = getattr(self, pre_hook)
                if replace_inputs:
                    args = internal_method(*args, **kwargs)
                    args = args if isinstance(args, Iterable) else (args,) 
                else:
                    internal_method(*args, **kwargs)
        if callable(pre_hook):
            pre_hook(self, *args, **kwargs)
        # Execute the original method
        result = original_method(self, *args, **kwargs)
        # Post-hook
        if post_hook:
            if callable(post_hook):
                post_hook(self, *args, **kwargs)
            elif isinstance(post_hook, str):
                internal_method = getattr(self, post_hook)
                if replace_outputs:
                    result = internal_method(result, *args, **kwargs)
                else:
                    internal_method(result, *args, **kwargs)
        return result
    return wrapped_method

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class dualmethod:
    """
        Decorator for instance/class method definition of methods under the same name.
    """
    def __init__(self, f_instance):
        self.f_instance = f_instance
        self.f_class = None

    def class_method(self, f_class):
        """
            Decorator to register the class-level behavior.
        """
        self.f_class = f_class
        return self

    def __get__(self, instance, owner):
        """
            This is the core of the descriptor protocol. Python calls this
            method whenever the attribute is accessed.
            
            Args:
                instance: The instance the method was called on, or None if
                        called on the class.
                owner: The class that owns the method.
        """
        if instance is None:
            # Called on the class
            if self.f_class is None:
                raise AttributeError('This method does not have a class-level implementation.')
            return functools.partial(self.f_class, owner)
        else:
            # Called on an instance
            return functools.partial(self.f_instance, instance)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################