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

DEFAULT_SPARKMODULE_PATH = 'spark.core.module.SparkModule'
DEFAULT_PAYLOAD_PATH = 'spark.core.payloads.SparkPayload'
DEFAULT_CONFIG_PATH = 'spark.core.config.BaseSparkConfig'
DEFAULT_INITIALIZER_PATH = 'spark.nn.initializers.base.Initializer'
DEFAULT_INITIALIZER_CONFIG_PATH = 'spark.nn.initializers.base.InitializerConfig'
DEFAULT_CFG_VALIDATOR_PATH = 'spark.core.config_validation.ConfigurationValidator'

def _is_spark_type(obj: tp.Any, type_name: str) -> bool:
    """
        Check if a given object is a subclass of a specific fully qualified type name.

        Notes:
            THIS METHOD IS INTENDED FOR INTERNAL USAGE ONLY.
            This function relies on matching the string representation of each class
            in the MRO with the target type name. It only works if 'obj' is an actual class.

        Args:
            obj (tp.Any): The class to check.
            type_name (str): The fully qualified name of the target type.
        Returns:
            bool, True if 'obj' is a subclass of the specified type, False otherwise.
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
        Check if an object instance is derived from a specific fully qualified type name.

        Notes:
            THIS METHOD IS INTENDED FOR INTERNAL USAGE ONLY.
            This function relies on matching the string representation of each class
            in the MRO with the target type name. It only works if 'obj' is an actual class.

        Args:
            obj (tp.Any): The instance to check.
            type_name (str): The fully qualified name of the target type.
        Returns:
            bool, True if the instance is derived from the specified type, False otherwise.
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
        Check if an object is a subclass of 'spark.nn.initializers.base.Initializer'.

        Args:
            obj (tp.Any): The class to check.
        Returns:
            bool, True if 'obj' is a subclass of 'Initializer', False otherwise.
    """
    return _is_spark_type(obj, DEFAULT_INITIALIZER_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_initializer_instance(obj: tp.Any) -> bool:
    """
        Check if an object instance is derived from 'spark.nn.initializers.base.Initializer'.

        Args:
            obj (tp.Any): The class to check.
        Returns:
            bool, True if 'obj' is an instance of 'Initializer', False otherwise.
    """
    return _is_spark_instance(obj, DEFAULT_INITIALIZER_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_initializer_config_type(obj: tp.Any) -> bool:
    """
        Check if an object is a subclass of 'spark.nn.initializers.base.InitializerConfig'.

        Args:
            obj (tp.Any): The class to check.
        Returns:
            bool, True if 'obj' is a subclass of 'InitializerConfig', False otherwise.
    """
    return _is_spark_type(obj, DEFAULT_INITIALIZER_CONFIG_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_payload_type(obj: tp.Any) -> bool:
    """
        Check if an object is a subclass of 'spark.core.payloads.SparkPayload'.

        Args:
            obj (tp.Any): The class to check.
        Returns:
            bool, True if 'obj' is a subclass of 'SparkPayload', False otherwise.
    """
    return _is_spark_type(obj, DEFAULT_PAYLOAD_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_payload_instance(obj: tp.Any) -> bool:
    """
        Check if an object instance is derived from 'spark.core.payloads.SparkPayload'.

        Args:
            obj (tp.Any): The instance to check.
        Returns:
            bool, True if the 'obj' is an instance of 'SparkPayload', False otherwise.
    """
    return _is_spark_instance(obj, DEFAULT_PAYLOAD_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_module_instance(obj: tp.Any) -> bool:
    """
        Check if an object is an instance of 'spark.core.module.SparkModule'.

        Args:
            obj (tp.Any): The class to check.
        Returns:
            bool, True if 'obj' is an instance of 'SparkModule', False otherwise.
    """
    return _is_spark_instance(obj, DEFAULT_SPARKMODULE_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_module_type(obj: tp.Any) -> bool:
    """
        Check if an object is a subclass of 'spark.core.module.SparkModule'.

        Args:
            obj (tp.Any): The class to check.
        Returns:
            bool, True if 'obj' is a subclass of 'SparkModule', False otherwise.
    """
    return _is_spark_type(obj, DEFAULT_SPARKMODULE_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_config_instance(obj: tp.Any) -> bool:
    """
        Check if an object instance is derived from DEFAULT_CONFIG_PATH.

        Args:
            obj (tp.Any): The instance to check.
        Returns:
            bool, True if the object is an instance of 'BaseSparkConfig', False otherwise.
    """
    return _is_spark_instance(obj, DEFAULT_CONFIG_PATH)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_config_type(obj: tp.Any) -> bool:
    """
        Check if an object a subclass of from DEFAULT_CONFIG_PATH.

        Args:
            obj (tp.Any): The instance to check.
        Returns:
            bool, True if the object is a subclass of 'BaseSparkConfig', False otherwise.
    """
    return _is_spark_type(obj, DEFAULT_CONFIG_PATH)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################