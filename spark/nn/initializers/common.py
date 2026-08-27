#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import jax
import jax.numpy as jnp 
import dataclasses as dc
import typing as tp
import spark.core.utils as utils
from spark.core.registry import register_initializer, register_config
from spark.core.config_validation import TypeValidator, ZeroOneValidator
from spark.nn.initializers.base import Initializer, InitializerConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

@register_config
class ConstantInitializerConfig(InitializerConfig):
    """
        Configuration for `ConstantInitializer`.

        Parameters
        ----------
        dtype : DTypeLike, default jnp.float16
            Dtype of the produced array.
        scale : int or float, default 1
            The value every entry takes.
    """
    __class_ref__: tp.ClassVar[str] = 'ConstantInitializer'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_initializer
class ConstantInitializer(Initializer):
    """
        Fills the array with one value: ``scale``.

        Parameters
        ----------
        config : ConstantInitializerConfig, optional
            Initializer configuration. Its fields may also be given as keyword arguments.
    """
    config: ConstantInitializerConfig

    def __call__(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """
            Returns an array filled with ``scale``.

            Parameters
            ----------
            key : jax.Array
                PRNG key. Unused, and accepted only to match the initializer signature.
            shape : tuple of int
                Shape of the array.

            Returns
            -------
            jax.Array
                The array, cast to ``dtype``.
        """
        array: jax.Array = self.config.scale * jnp.ones(shape, dtype=self.config.dtype)
        return array.astype(self.config.dtype)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class UniformInitializerConfig(InitializerConfig):
    """
        Configuration for `UniformInitializer`.

        Parameters
        ----------
        dtype : DTypeLike, default jnp.float16
            Dtype of the produced array.
        scale : int or float, default 1
            Upper end of the range drawn from.
        min_value : int or float or None, default None
            Lower clip applied after drawing.
        max_value : int or float or None, default None
            Upper clip applied after drawing.
    """
    __class_ref__: tp.ClassVar[str] = 'UniformInitializer'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_initializer
class UniformInitializer(Initializer):
    """
        Draws every entry uniformly from ``[0, scale)``.

        Parameters
        ----------
        config : UniformInitializerConfig, optional
            Initializer configuration. Its fields may also be given as keyword arguments.

        Notes
        -----
        ``min_value`` and ``max_value`` clip the result, so they narrow the range rather than
        shift it.

        See Also
        --------
        SparseUniformInitializer : The same draw with a fraction of the entries zeroed.
    """
    config: UniformInitializerConfig

    def __call__(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """
            Draws every entry uniformly from ``[0, scale)``.

            Parameters
            ----------
            key : jax.Array
                PRNG key.
            shape : tuple of int
                Shape of the array to draw.

            Returns
            -------
            jax.Array
                The drawn array, clipped to ``[min_value, max_value]`` and cast to ``dtype``.
        """
        array = self.config.scale * jax.random.uniform(key, shape)
        # Clip Min-Max
        array = jnp.clip(array, min=self.config.min_value, max=self.config.max_value)
        return array.astype(self.config.dtype)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class SparseUniformInitializerConfig(UniformInitializerConfig):
    """
        Configuration for `SparseUniformInitializer`.

        Parameters
        ----------
        dtype : DTypeLike, default jnp.float16
            Dtype of the produced array.
        scale : int or float, default 1
            Upper end of the range drawn from.
        density : float, default 0.2
            Expected fraction of non-zero entries.
        min_value : int or float or None, default None
            Lower clip applied after drawing.
        max_value : int or float or None, default None
            Upper clip applied after drawing.
    """
    __class_ref__: tp.ClassVar[str] = 'SparseUniformInitializer'

    density: float = dc.field(
        default = 0.2, 
        metadata = {
            'validators': [
                TypeValidator,
                ZeroOneValidator,
            ],
            'description': 'Expected ratio of non-zero entries in the output array.',
        })

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_initializer
class SparseUniformInitializer(UniformInitializer):
    """
        Draws every entry uniformly, then zeroes a fraction of them.

        Each entry is kept with probability ``density`` and set to zero otherwise, so the number
        of non-zero entries varies between draws around its expected value.

        Parameters
        ----------
        config : SparseUniformInitializerConfig, optional
            Initializer configuration. Its fields may also be given as keyword arguments.

        Notes
        -----
        The zeroing is applied before the clip, so a ``min_value`` above zero fills the zeroed
        entries back in.

        See Also
        --------
        NormalizedSparseUniformInitializer : The same draw, normalized along chosen axes.
    """
    config: SparseUniformInitializerConfig

    def __call__(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """
            Draws every entry uniformly, then zeroes a fraction of them.

            Parameters
            ----------
            key : jax.Array
                PRNG key. Split once, for the values and for the zeroing mask.
            shape : tuple of int
                Shape of the array to draw.

            Returns
            -------
            jax.Array
                The drawn array, with each entry kept with probability ``density``, clipped to
                ``[min_value, max_value]`` and cast to ``dtype``.
        """
        key1, key2 = jax.random.split(key, 2)
        # Get uniform array
        array = self.config.scale * jax.random.uniform(key1, shape)
        # Zero mask
        mask = jax.random.uniform(key2, shape, dtype=jnp.float16) < self.config.density
        array = jnp.where(mask, array, 0)
        # Clip Min-Max
        array = jnp.clip(array, min=self.config.min_value, max=self.config.max_value)
        return array.astype(self.config.dtype)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_config
class NormalizedSparseUniformInitializerConfig(SparseUniformInitializerConfig):
    """
        Configuration for `NormalizedSparseUniformInitializer`.

        Parameters
        ----------
        dtype : DTypeLike, default jnp.float16
            Dtype of the produced array. Must be a float type.
        scale : int or float, default 1
            Factor applied after normalization, so each normalized group sums to ``scale``.
        density : float, default 0.2
            Expected fraction of non-zero entries.
        norm_axes : tuple of int, default (0,)
            Axes the sums are taken over. Set by the module that requests the array; a synapse
            passes its postsynaptic axes.
        min_value : int or float or None, default None
            Lower clip applied after normalization.
        max_value : int or float or None, default None
            Upper clip applied after normalization.
    """
    __class_ref__: tp.ClassVar[str] = 'NormalizedSparseUniformInitializer'

    norm_axes: tuple[int, ...] | None = dc.field(
        default = (0,), 
        metadata = {
            'validators': [
                TypeValidator,
            ],
            'description': 'Axes used to normalize the output array over. Note: This attribute is automatically managed.',
        }
    )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@register_initializer
class NormalizedSparseUniformInitializer(SparseUniformInitializer):
    """
        Sparse uniform draw normalized along chosen axes.

        `SparseUniformInitializer` followed by a division by the sum over ``norm_axes``, so every
        group along those axes sums to ``scale``. For a synaptic kernel this fixes the total input
        a postsynaptic unit receives, independent of how many presynaptic units survived the
        sparsification.

        Parameters
        ----------
        config : NormalizedSparseUniformInitializerConfig, optional
            Initializer configuration. Its fields may also be given as keyword arguments.

        Raises
        ------
        ValueError
            If the shape is one dimensional, or if ``norm_axes`` holds a repeated or
            out-of-range axis.
        TypeError
            If ``dtype`` is not a float type.

        Notes
        -----
        A group that sums to zero is left as it is rather than divided.

        See Also
        --------
        SparseUniformInitializer : The same draw, without normalization.
    """
    config: NormalizedSparseUniformInitializerConfig

    def __call__(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        # Normalize
        """
            Draws a sparse uniform array normalized along ``norm_axes``.

            Parameters
            ----------
            key : jax.Array
                PRNG key. Split once, for the values and for the zeroing mask.
            shape : tuple of int
                Shape of the array to draw. Must have two dimensions or more.

            Returns
            -------
            jax.Array
                The drawn array, with every group along ``norm_axes`` summing to ``scale``, clipped
                to ``[min_value, max_value]`` and cast to ``dtype``.

            Raises
            ------
            ValueError
                If ``shape`` is one dimensional, or if ``norm_axes`` holds a repeated or
                out-of-range axis.
            TypeError
                If ``dtype`` is not a float type.
        """
        num_dims = len(shape)
        # Sanity checks  
        if not num_dims > 1:
            raise ValueError(
                f'Normalization is only supported for arrays of dimension 2 or larger but got \"shape\": {shape}.'
            )
        if not utils.is_float(self.config.dtype):
            raise TypeError(
                f'Normalization is only possible for float \"dtype\", but got: \"{self.config.dtype}\".'
            )
        if any([(ax < 0) and (ax >= num_dims) for ax in self.config.norm_axes]):
            raise ValueError(
                f'Expected all indices of \"norm_axes\" to be in the set {{0, ..., {num_dims-1}}}, but got: \"{self.config.norm_axes}\".'
            )
        if len(set(ax for ax in self.config.norm_axes)) != len(self.config.norm_axes):
            raise ValueError(
                f'Expected all indices of \"norm_axes\" to be unique, but got: \"{self.config.norm_axes}\".'
            )
        # Get sparse array
        key1, key2 = jax.random.split(key, 2)
        # Get uniform array
        array = jax.random.uniform(key1, shape)
        # Zero mask
        mask = jax.random.uniform(key2, shape, dtype=jnp.float16) < self.config.density
        array = jnp.where(mask, array, 0)
        # Normalize axes labes
        all_labels = utils.get_axes_einsum_labels([i for i in range(len(shape))])
        norm_labels = utils.get_axes_einsum_labels(self.config.norm_axes)
        # Normalize
        norm = jnp.einsum(f'{all_labels}->{norm_labels}', array)
        norm = jnp.where(norm != 0, 1/norm, 1)
        array = jnp.einsum(f'{all_labels},{norm_labels}->{all_labels}', array, norm)
        # Clip Min-Max
        array = jnp.clip(self.config.scale * array, min=self.config.min_value, max=self.config.max_value)
        return array.astype(self.config.dtype)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################