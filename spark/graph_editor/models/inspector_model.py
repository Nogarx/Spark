#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import typing as tp
import dataclasses as dc
from PySide6.QtCore import QObject, Signal
from spark.core.specs import ModuleSpecs
from spark.core.config import SparkConfig
from spark.nn.initializers import InitializerConfig

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class ConfigNode(QObject):
    """
        Base class for all configuration nodes in the inspector state model.
        This provides a generic, observable tree structure that the UI can bind to,
        isolating the editor from the complex nested dataclasses and ModuleSpecs.
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
    def __init__(self, name: str, value: tp.Any, type_hint: type, parent: QObject = None) -> None:
        super().__init__(name, parent)
        self._value = value
        self.type_hint = type_hint

    @property
    def value(self) -> tp.Any:
        return self._value

    @value.setter
    def value(self, new_val) -> None:
        if self._value != new_val:
            self._value = new_val
            self.value_changed.emit(self, new_val)

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
        # Propagate child value changes up so the inspector can react if needed
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

def parse_object_to_state(name: str, obj: tp.Any, type_hint: type = None, metadata: dict = None, parent: QObject = None) -> ConfigNode:
    """
        Recursively parses a Python object (SparkConfig, Dataclass, ModuleSpecs, List, or Primitive) into a UI-bindable ConfigNode tree.
    """
    metadata = metadata or {}
    
    # Handle ModuleSpecs
    if isinstance(obj, ModuleSpecs):
        node = ConfigGroupNode(name, type(obj), parent)
        node.metadata = metadata
        node.add_child(ConfigValueNode('name', obj.name, str, parent=node))
        node.add_child(parse_object_to_state('config', obj.config, type_hint=None, parent=node))

    # Handle Primitive Fields holding an InitializerConfig
    elif metadata.get('allows_init', False) and isinstance(obj, InitializerConfig):
        node = ConfigValueNode(name, obj, type_hint or type(obj), parent)
        node.metadata = metadata
        node._is_initializer_active = True

    # Handle SparkConfig
    elif isinstance(obj, SparkConfig):
        node = ConfigGroupNode(name, type(obj), parent)
        node.metadata = metadata
        for field in dc.fields(obj):
            value = getattr(obj, field.name)
            # Use field.type for type_hint if available, and pass field.metadata
            child_node = parse_object_to_state(
                field.name, 
                value, 
                type_hint=field.type, 
                metadata=dict(field.metadata), 
                parent=node
            )
            # Check for dummy error method and bind to child node
            if not isinstance(obj, type) and hasattr(obj, 'get_field_errors'):
                errs = obj.get_field_errors(field.name)
                if errs:
                    child_node.errors = errs
            node.add_child(child_node)
        
    # Handle Lists
    elif isinstance(obj, list):
        # Attempt to infer the item type
        item_type = type(obj[0]) if len(obj) > 0 else tp.Any
        node = ConfigListNode(name, item_type, parent)
        node.metadata = metadata
        for i, item in enumerate(obj):
            # The name "[i]" is useful for UI labels, but is ignored during to_python() list reconstruction
            node.add_child(parse_object_to_state(f"[{i}]", item, type_hint=item_type, parent=node))

    # Handle Primitives (int, float, str, bool, etc.)
    else:
        node = ConfigValueNode(name, obj, type_hint or type(obj), parent)
        node.metadata = metadata
        
    return node

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################