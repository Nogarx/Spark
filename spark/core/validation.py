#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

from typing import Any

# TODO: These methods are useful to prevent some circular imports, specially  
# with the parser and registry, but there should be a better way to validate.

# NOTE: This methods are only intended for internal usage given its brittleness.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

DEFAULT_SPARKMODULE_PATH = 'spark.core.module.SparkModule'
DEFAULT_PAYLOAD_PATH = 'spark.core.payloads.SparkPayload'

def _is_spark_type(obj: Any, type_name: str) -> bool:
    """
        Check if a given object is a subclass of a specific fully qualified type name.

        Notes:
            THIS METHOD IS INTENDED FOR INTERNAL USAGE ONLY.
            This function relies on matching the string representation of each class
            in the MRO with the target type name. It only works if 'obj' is an actual class.

        Args:
            obj (Any): The class to check.
            type_name (str): The fully qualified name of the target type.
        Returns:
            bool: True if 'obj' is a subclass of the specified type, False otherwise.
    """
    if isinstance(obj, type):
        for sub_cls in obj.__mro__:
            sub_cls_path = str(sub_cls).split("'")[1]
            if sub_cls_path == type_name:
                return True
    return False


def _is_spark_instance(obj: Any, type_name: str) -> bool:
    """
        Check if an object instance is derived from a specific fully qualified type name.

        Notes:
            THIS METHOD IS INTENDED FOR INTERNAL USAGE ONLY.
            This function relies on matching the string representation of each class
            in the MRO with the target type name. It only works if 'obj' is an actual class.

        Args:
            obj (Any): The instance to check.
            type_name (str): The fully qualified name of the target type.
        Returns:
            bool: True if the instance is derived from the specified type, False otherwise.
    """
    if not isinstance(obj, type):
        for sub_cls in type(obj).__mro__:
            sub_cls_path = str(sub_cls).split("'")[1]
            if sub_cls_path == type_name:
                return True
    return False


def _is_payload_type(obj: Any) -> bool:
    """
        Check if a object is a subclass of 'spark.core.payloads.SparkPayload'.

        Args:
            obj (Any): The class to check.
        Returns:
            bool: True if 'obj' is a subclass of 'SparkPayload', False otherwise.
    """
    return _is_spark_type(obj, DEFAULT_PAYLOAD_PATH)


def _is_payload_instance(obj: Any) -> bool:
    """
        Check if an object instance is derived from 'spark.core.payloads.SparkPayload'.

        Args:
            obj (Any): The instance to check.
        Returns:
            bool: True if the 'obj' is an instance of 'SparkPayload', False otherwise.
    """
    return _is_spark_instance(obj, DEFAULT_PAYLOAD_PATH)


def _is_module_type(obj: Any) -> bool:
    """
        Check if a object is a subclass of 'spark.core.module.SparkModule'.

        Args:
            obj (Any): The class to check.
        Returns:
            bool: True if 'obj' is a subclass of 'SparkModule', False otherwise.
    """
    return _is_spark_type(obj, DEFAULT_SPARKMODULE_PATH)


def _is_module_instance(obj: Any) -> bool:
    """
        Check if an object instance is derived from 'spark.core.module.SparkModule'.

        Args:
            obj (Any): The instance to check.
        Returns:
            bool: True if the object is an instance of 'SparkModule', False otherwise.
    """
    return _is_spark_instance(obj, DEFAULT_SPARKMODULE_PATH)


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################