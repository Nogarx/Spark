#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import logging
import typing as tp
import dataclasses as dc
from PySide6.QtCore import QObject, Signal
from spark.core.specs import ModuleSpecs
from spark.core.config import SparkConfig
from spark.nn.initializers import InitializerConfig
from spark.graph_editor.models.config_types import FieldKind, classify, type_tokens, is_optional, values_equal

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ConfigNode(QObject):
    """
        Base class for all configuration nodes in the inspector state model.

        Observable tree structure the UI binds to, wrapping the nested dataclasses and ModuleSpecs.
    """
    value_changed = Signal(object, object) 
    errors_changed = Signal(object, list)
    inheritance_changed = Signal(object, bool)
    initializer_changed = Signal(object, bool)

    def __init__(self, name: str, parent: QObject = None) -> None:
        super().__init__(parent)
        self.name = name
        self._errors: list[str] = []
        self._is_inherited: bool = False
        self._is_initializer_active: bool = False
        self.metadata: dict = {}

    @property
    def errors(self) -> list[str]:
        return self._errors
        
    @errors.setter
    def errors(self, value: list[str]) -> None:
        if self._errors != value:
            self._errors = value
            self.errors_changed.emit(self, value)
            
    @property
    def is_inherited(self) -> bool:
        return self._is_inherited
        
    @is_inherited.setter
    def is_inherited(self, value: bool) -> None:
        if self._is_inherited != value:
            self._is_inherited = value
            self.inheritance_changed.emit(self, value)
            
    @property
    def is_initializer_active(self) -> bool:
        return self._is_initializer_active
        
    @is_initializer_active.setter
    def is_initializer_active(self, value: bool) -> None:
        if self._is_initializer_active != value:
            self._is_initializer_active = value
            self.initializer_changed.emit(self, value)

    def to_python(self) -> tp.Any:
        """
            Reconstructs the underlying Python object (dataclass, list, or primitive).
        """
        raise NotImplementedError

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ConfigValueNode(ConfigNode):
    """
        Represents a primitive or simple value (int, float, str, bool, enum, etc.).
    """
    def __init__(self, name: str, value: tp.Any, type_hint: type, parent: QObject = None, field: dc.Field | None = None) -> None:
        super().__init__(name, parent)
        self._value = value
        self.type_hint = type_hint
        # Originating dataclass field. Required to run the field validators declared by the configuration.
        self.field = field

    @property
    def value(self) -> tp.Any:
        return self._value

    @value.setter
    def value(self, new_val) -> None:
        # NOTE: Arrays return element-wise comparisons, so a plain "!=" cannot be used.
        if not values_equal(self._value, new_val):
            self._value = new_val
            self.revalidate()
            self.value_changed.emit(self, new_val)

    @property
    def is_required(self) -> bool:
        """
            True if the underlying configuration field defines neither a default nor a default factory.
        """
        if self.field is None:
            return False
        return self.field.default is dc.MISSING and self.field.default_factory is dc.MISSING

    def revalidate(self) -> None:
        """
            Recomputes the error list of this node from the value and the field validators.
        """
        self.errors = collect_field_errors(
            self.name, self._value, self.metadata, self.field, self.is_required, self.type_hint
        )

    def to_python(self) -> tp.Any:
        return self._value

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ConfigGroupNode(ConfigNode):
    """
        Represents a nested object, typically a SparkConfig, or ModuleSpecs.
    """
    def __init__(self, name: str, class_ref: type, parent: QObject = None) -> None:
        super().__init__(name, parent)
        self.class_ref = class_ref
        self.children: list[ConfigNode] = []

    def add_child(self, child: ConfigNode) -> None:
        child.setParent(self)
        self.children.append(child)
        # Propagate child value changes up, so the inspector can react to them.
        child.value_changed.connect(self._on_child_value_changed)

    def _on_child_value_changed(self, node, new_val) -> None:
        self.value_changed.emit(node, new_val)

    def to_python(self) -> tp.Any:
        kwargs = {child.name: child.to_python() for child in self.children}
        
        # ModuleSpecs reconstruction
        if issubclass(self.class_ref, ModuleSpecs):
            return ModuleSpecs(
                name=kwargs.get('name'),
                config=kwargs.get('config')
            )
            
        # Standard dataclass/config reconstruction
        return self.class_ref(**kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class ConfigListNode(ConfigNode):
    """
        Represents a list of items, such as list[ModuleSpecs] or list[int].
    """
    def __init__(self, name: str, item_type: type, parent: QObject = None) -> None:
        super().__init__(name, parent)
        self.item_type = item_type
        self.children: list[ConfigNode] = []

    def add_child(self, child: ConfigNode) -> None:
        child.setParent(self)
        self.children.append(child)
        child.value_changed.connect(self._on_child_value_changed)

    def _on_child_value_changed(self, node, new_val) -> None:
        self.value_changed.emit(node, new_val)

    def to_python(self) -> list:
        return [
            child.to_python() for child in self.children
        ]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def collect_field_errors(
        name: str,
        value: tp.Any,
        metadata: dict | None = None,
        field: dc.Field | None = None,
        is_required: bool = False,
        type_hint: tp.Any = None,
    ) -> list[str]:
    """
        Runs the validators declared by a configuration field against a value.

        Parameters
        ----------
        name : str
            Field name, used to build the messages.
        value : Any
            Current value of the field.
        metadata : dict, optional
            Field metadata. Validators are read from the "validators" entry.
        field : dataclasses.Field, optional
            Originating dataclass field. Validators are constructed from it.
        is_required : bool, default False
            True if the field defines neither a default nor a default factory.
        type_hint : Any, optional
            Field annotation. Used to detect fields that accept None.

        Returns
        -------
        list of str
            The collected error messages.
    """
    metadata = metadata or {}
    errors: list[str] = []
    if value is None:
        # NOTE: A partial configuration leaves unset fields as None. Missing values are not reported while
        # editing, that check belongs to the export validation. Only actual values are validated here.
        return errors
    # NOTE: Validators are instantiated from the dataclass field, so they cannot run without it.
    if field is None:
        return errors
    for validator_cls in (metadata.get('validators') or []):
        try:
            validator_cls(field).validate(value)
        except Exception as error:
            errors.append(str(error))
    return errors

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _is_container_of_configs(obj: tp.Any) -> bool:
    """
        True if the object is a non-empty collection of ModuleSpecs/SparkConfig instances.
    """
    if not isinstance(obj, (list, tuple)):
        return False
    if len(obj) == 0:
        return False
    return all(isinstance(item, (ModuleSpecs, SparkConfig)) for item in obj)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def parse_object_to_state(
        name: str,
        obj: tp.Any,
        type_hint: type = None,
        metadata: dict = None,
        parent: QObject = None,
        field: dc.Field | None = None,
    ) -> ConfigNode:
    """
        Parses a Python object (SparkConfig, dataclass, ModuleSpecs, list or primitive) into a ConfigNode tree.
    """
    metadata = metadata or {}

    if isinstance(obj, ModuleSpecs):
        node = ConfigGroupNode(name, type(obj), parent)
        node.metadata = metadata
        node.add_child(ConfigValueNode('name', obj.name, str, parent=node))
        node.add_child(parse_object_to_state('config', obj.config, type_hint=None, parent=node))

    elif metadata.get('allows_init', False) and isinstance(obj, InitializerConfig):
        node = ConfigValueNode(name, obj, type_hint or type(obj), parent, field=field)
        node.metadata = metadata
        node._is_initializer_active = True

    elif isinstance(obj, SparkConfig):
        node = ConfigGroupNode(name, type(obj), parent)
        node.metadata = metadata
        for child_field in dc.fields(obj):
            value = getattr(obj, child_field.name, None)
            child_node = parse_object_to_state(
                child_field.name,
                value,
                type_hint=child_field.type,
                metadata=dict(child_field.metadata),
                parent=node,
                field=child_field,
            )
            node.add_child(child_node)

    # NOTE: SparkConfigMeta crystallizes every mutable iterable into a tuple, so ModuleSpecs collections
    # reach the editor as tuples. An empty collection is rendered as a list when the field declares one.
    elif _is_container_of_configs(obj) or (
            isinstance(obj, (list, tuple)) and classify(type_hint, metadata.get('valid_types')) is FieldKind.MODULE_SPECS
        ):
        item_type = type(obj[0]) if len(obj) > 0 else ModuleSpecs
        node = ConfigListNode(name, item_type, parent)
        node.metadata = metadata
        for i, item in enumerate(obj):
            # The name "[i]" is used for UI labels and ignored during to_python() list reconstruction.
            node.add_child(parse_object_to_state(f"[{i}]", item, type_hint=item_type, parent=node))

    # Handle Primitives (int, float, str, bool, tuple[int, ...], arrays, etc.)
    else:
        node = ConfigValueNode(name, obj, type_hint or type(obj), parent, field=field)
        node.metadata = metadata

    if isinstance(node, ConfigValueNode):
        node.revalidate()
    return node

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################