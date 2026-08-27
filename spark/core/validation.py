#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import typing as tp
import jax.numpy as jnp

# TODO: These methods are useful to prevent some circular imports, specially  
# with the parser and registry, but there should be a better way to validate.

# NOTE: This methods are only intended for internal usage given its brittleness.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

DEFAULT_SPARK_MODULE_PATH = 'spark.core.module.SparkModule'
DEFAULT_SPARK_CONTROLLER_PATH = 'spark.nn.controllers.base.Controller'
DEFAULT_SPARK_NEURON_PATH = 'spark.nn.controllers.neuron.Neuron'
DEFAULT_INTERFACE_PATH = 'spark.nn.interfaces.base.Interface'
DEFAULT_PAYLOAD_PATH = 'spark.core.payloads.SparkPayload'
DEFAULT_CONFIG_PATH = 'spark.core.config.SparkConfig'
DEFAULT_INITIALIZER_PATH = 'spark.nn.initializers.base.Initializer'
DEFAULT_INITIALIZER_CONFIG_PATH = 'spark.nn.initializers.base.InitializerConfig'
DEFAULT_CFG_VALIDATOR_PATH = 'spark.core.config_validation.ConfigurationValidator'

def _is_spark_type(obj: tp.Any, type_name: str) -> bool:
    """
        Whether a class is a subclass of the type named by a qualified name.

        Parameters
        ----------
        obj : Any
            Class to check.
        type_name : str
            Fully qualified name of the target type.

        Returns
        -------
        bool
            False for anything that is not a class.

        Notes
        -----
        Internal. Matches ``type_name`` against the string form of every class in the MRO, which
        is what lets a module test for a type it cannot import without a circular import.
    """
    if isinstance(obj, type):
        for sub_cls in obj.__mro__:
            sub_cls_path = str(sub_cls).split("'")[1]
            if sub_cls_path == type_name:
                return True
    return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_spark_instance(obj: tp.Any, type_name: str) -> bool:
    """
        Whether an instance derives from the type named by a qualified name.

        Parameters
        ----------
        obj : Any
            Instance to check.
        type_name : str
            Fully qualified name of the target type.

        Returns
        -------
        bool

        Notes
        -----
        Internal. Matches ``type_name`` against the string form of every class in the MRO, which
        is what lets a module test for a type it cannot import without a circular import.
    """
    if not isinstance(obj, type):
        for sub_cls in type(obj).__mro__:
            sub_cls_path = str(sub_cls).split("'")[1]
            if sub_cls_path == type_name:
                return True
    return False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_initializer_type(obj: tp.Any) -> bool:
    """
        Whether an object is a subclass of `Initializer`.

        Parameters
        ----------
        obj : Any
            Class to check.

        Returns
        -------
        bool
            False for anything that is not a class.

        Notes
        -----
        Internal. Matches the qualified name against the MRO by string, so it answers for a class
        the caller cannot import without a circular import.
    """
    return _is_spark_type(obj, DEFAULT_INITIALIZER_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_initializer_instance(obj: tp.Any) -> bool:
    """
        Whether an object is an instance of `Initializer`.

        Parameters
        ----------
        obj : Any
            Instance to check.

        Returns
        -------
        bool

        Notes
        -----
        Internal. Matches the qualified name against the MRO by string, so it answers for a class
        the caller cannot import without a circular import.
    """
    return _is_spark_instance(obj, DEFAULT_INITIALIZER_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_initializer_config_type(obj: tp.Any) -> bool:
    """
        Whether an object is a subclass of `InitializerConfig`.

        Parameters
        ----------
        obj : Any
            Class to check.

        Returns
        -------
        bool
            False for anything that is not a class.

        Notes
        -----
        Internal. Matches the qualified name against the MRO by string, so it answers for a class
        the caller cannot import without a circular import.
    """
    return _is_spark_type(obj, DEFAULT_INITIALIZER_CONFIG_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_payload_type(obj: tp.Any) -> bool:
    """
        Whether an object is a subclass of `SparkPayload`.

        Parameters
        ----------
        obj : Any
            Class to check.

        Returns
        -------
        bool
            False for anything that is not a class.

        Notes
        -----
        Internal. Matches the qualified name against the MRO by string, so it answers for a class
        the caller cannot import without a circular import.
    """
    return _is_spark_type(obj, DEFAULT_PAYLOAD_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_payload_instance(obj: tp.Any) -> bool:
    """
        Whether an object is an instance of `SparkPayload`.

        Parameters
        ----------
        obj : Any
            Instance to check.

        Returns
        -------
        bool

        Notes
        -----
        Internal. Matches the qualified name against the MRO by string, so it answers for a class
        the caller cannot import without a circular import.
    """
    return _is_spark_instance(obj, DEFAULT_PAYLOAD_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_module_instance(obj: tp.Any) -> bool:
    """
        Whether an object is an instance of `SparkModule`.

        Parameters
        ----------
        obj : Any
            Instance to check.

        Returns
        -------
        bool

        Notes
        -----
        Internal. Matches the qualified name against the MRO by string, so it answers for a class
        the caller cannot import without a circular import.
    """
    return _is_spark_instance(obj, DEFAULT_SPARK_MODULE_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_module_type(obj: tp.Any) -> bool:
    """
        Whether an object is a subclass of `SparkModule`.

        Parameters
        ----------
        obj : Any
            Class to check.

        Returns
        -------
        bool
            False for anything that is not a class.

        Notes
        -----
        Internal. Matches the qualified name against the MRO by string, so it answers for a class
        the caller cannot import without a circular import.
    """
    return _is_spark_type(obj, DEFAULT_SPARK_MODULE_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_controller_instance(obj: tp.Any) -> bool:
    """
        Whether an object is an instance of `Controller`.

        Parameters
        ----------
        obj : Any
            Instance to check.

        Returns
        -------
        bool

        Notes
        -----
        Internal. Matches the qualified name against the MRO by string, so it answers for a class
        the caller cannot import without a circular import.
    """
    return _is_spark_instance(obj, DEFAULT_SPARK_CONTROLLER_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_controller_type(obj: tp.Any) -> bool:
    """
        Whether an object is a subclass of `Controller`.

        Parameters
        ----------
        obj : Any
            Class to check.

        Returns
        -------
        bool
            False for anything that is not a class.

        Notes
        -----
        Internal. Matches the qualified name against the MRO by string, so it answers for a class
        the caller cannot import without a circular import.
    """
    return _is_spark_type(obj, DEFAULT_SPARK_CONTROLLER_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_config_instance(obj: tp.Any) -> bool:
    """
        Whether an object is an instance of `SparkConfig`.

        Parameters
        ----------
        obj : Any
            Instance to check.

        Returns
        -------
        bool

        Notes
        -----
        Internal. Matches the qualified name against the MRO by string, so it answers for a class
        the caller cannot import without a circular import.
    """
    return _is_spark_instance(obj, DEFAULT_CONFIG_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_config_type(obj: tp.Any) -> bool:
    """
        Whether an object is a subclass of `SparkConfig`.

        Parameters
        ----------
        obj : Any
            Class to check.

        Returns
        -------
        bool
            False for anything that is not a class.

        Notes
        -----
        Internal. Matches the qualified name against the MRO by string, so it answers for a class
        the caller cannot import without a circular import.
    """
    return _is_spark_type(obj, DEFAULT_CONFIG_PATH)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################