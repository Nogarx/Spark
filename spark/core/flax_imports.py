#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import flax.nnx as nnx
import typing as tp
import typing_extensions as tpe
from jax._src.pjit import JitWrapped
from flax.typing import Missing, MISSING
from flax.nnx.transforms.autodiff import DiffState, AxisName
from flax.nnx import filterlib
from flax.nnx.graph import GraphDef, GraphState
from flax.nnx.variablelib import VariableState

A = tp.TypeVar('A')

# NOTE: Currently this code is just a shortcut of all the basic Flax's LAX methods. 
# Its only purpose is to reduce imports for the final user.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def data(value: A, /) -> A:
    return nnx.data(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def grad(f: tp.Callable[..., tp.Any] | Missing = MISSING,
         *,
         argnums: int | DiffState | tp.Sequence[int | DiffState] = 0,
         has_aux: bool = False,
         holomorphic: bool = False,
         allow_int: bool = False,
         reduce_axes: tp.Sequence[AxisName] = (),
         ) -> (tp.Callable[..., tp.Any] | tp.Callable[[tp.Callable[..., tp.Any]], tp.Callable[..., tp.Any]] ):
    """
        Wrapper around flax.nnx.grad to simply imports.
    """
    return nnx.grad(
        f, 
        argnums=argnums, 
        has_aux=has_aux, 
        holomorphic=holomorphic, 
        allow_int=allow_int, 
        reduce_axes=reduce_axes
    )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def jit(
        fun: tp.Callable[..., tp.Any] | type[Missing] = Missing,
        *,
        in_shardings: tp.Any = None,
        out_shardings: tp.Any = None,
        static_argnums: int | tp.Sequence[int] | None = None,
        static_argnames: str | tp.Iterable[str] | None = None,
        donate_argnums: int | tp.Sequence[int] | None = None,
        donate_argnames: str | tp.Iterable[str] | None = None,
        keep_unused: bool = False,
        device: jax.Device | None = None,
        backend: str | None = None,
        inline: bool = False,
        abstracted_axes: tp.Any | None = None,
        ) -> JitWrapped | tp.Callable[[tp.Callable[..., tp.Any]], JitWrapped]:
    """
        Wrapper around flax.nnx.jit to simply imports.
    """
    return nnx.jit(
        fun, 
        in_shardings=in_shardings, 
        out_shardings=out_shardings, 
        static_argnums=static_argnums, 
        static_argnames=static_argnames, 
        donate_argnums=donate_argnums, 
        donate_argnames=donate_argnames, 
        keep_unused=keep_unused, 
        device=device, 
        backend=backend, 
        inline=inline, 
        abstracted_axes=abstracted_axes
    )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def eval_shape(f: tp.Callable[..., A],
               *args: tp.Any,
               **kwargs: tp.Any,
               ) -> A:
    """
        Wrapper around flax.nnx.eval_shape to simply imports.
    """
    return nnx.eval_shape(f,
                          *args,
                          **kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def split(node: A, 
          *filters: filterlib.Filter
          ) -> tuple[GraphDef[A], GraphState | VariableState, tpe.Unpack[tuple[GraphState | VariableState, ...]],]:
    """
        Wrapper around flax.nnx.split to simply imports.
    """
    return nnx.split(
        node,
        *filters
    )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def merge(graphdef: GraphDef[A],
          state: tp.Any,
          /,
          *states: tp.Any, 
          ) -> A:
    """
        Wrapper around flax.nnx.merge to simply imports.
    """
    return nnx.merge(graphdef,
                     state,
                     *states)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

