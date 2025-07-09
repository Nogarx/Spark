#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import abc
import jax
import numpy as np
from dataclasses import dataclass

# NOTE: Direct jax.Array subclassing is not supported. Currently the best approach is to define __jax_array__ 
# (https://docs.jax.dev/en/latest/jep/28661-jax-array-protocol.html). However it is probable that such approach will be deprecated
# in the future (https://github.com/jax-ml/jax/issues/26764#event-16480127978). This makes creating a full fledge section of variables
# quite complicated and not worth it at this moment.
# The best we can do right now is go for a Level 1 approach (polymorphic inputs) plus a manual override of relevant operations.
# The goal eventually is get rid of all the .value and other things that may be annoying for the user in a more graceful and robust manner.

# TODO: Several parts of this code need to be further validated. 

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SparkPayload(abc.ABC):
    """
        Abstract payload definition to validate exchanges between SparkModule's.
    """

    def tree_flatten(self):
        children = ()
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ValueSparkPayload(SparkPayload, abc.ABC):
    """
        Abstract payload definition to single value payloads.
    """
    value: jax.Array

    def tree_flatten(self):
        children = (self.value,)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (value,) = children
        return cls(value=value,)

    def __jax_array__(self) -> jax.Array: 
        return self.value
    
    def __array__(self, dtype=None) -> jax.Array: 
        return np.array(self.value).astype(dtype if dtype else self.value.dtype)


#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Delay registry import to prevent circular import.
from spark.core.registry import register_payload

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
@register_payload
class SpikeArray(ValueSparkPayload):
    """
        Representation of a collection of spike events.
    """
    pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
@register_payload
class CurrentArray(ValueSparkPayload):
    """
        Representation of a collection of currents.
    """
    pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
@register_payload
class PotentialArray(ValueSparkPayload):
    """
        Representation of a collection of membrane potentials.
    """
    pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
@register_payload
class BooleanMask(ValueSparkPayload):
    """
        Representation of an inhibitory boolean mask.
    """
    pass
        
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
@register_payload
class IntegerMask(ValueSparkPayload):
    """
        Representation of an integer mask.
    """
    pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
@register_payload
class FloatArray(ValueSparkPayload):
    """
        Representation of a float array.
    """
    pass
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
@register_payload
class IntegerArray(ValueSparkPayload):
    """
        Representation of an integer array.
    """
    pass
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

