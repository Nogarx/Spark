#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spark.graph_editor.models.graph_model import GraphModel
    from spark.graph_editor.models.controller_profile import ControllerProfile
    from spark.core.config import SparkConfig

import copy
import logging
import typing as tp
import dataclasses as dc

from spark.core.specs import ModuleSpecs, PortMap
from spark.graph_editor.models.node_model import NodeModel, SourceNodeModel, SinkNodeModel, SelfPropertyNodeModel
from spark.graph_editor.models.port_model import PortModel
from spark.graph_editor.models.config_types import type_tokens, is_optional

logger = logging.getLogger('spark')

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# NOTE: This is the inverse of model_import: it turns what is on the canvas back into the configuration of a
# controller. The editor deliberately allows half finished graphs, so the translation never refuses to run:
# it reports what is missing and still produces the best configuration it can. Whether those problems matter
# is decided by the caller, a session accepts them and a model does not.

_CALL_ORIGIN = '__call__'
_SELF_ORIGIN = '__self__'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass
class ExportedGraph:
    """
        Result of translating a graph back into a controller configuration.
    """
    config: tp.Any = None
    specs: list[ModuleSpecs] = dc.field(default_factory=list)
    layout: dict[str, tuple[float, float]] = dc.field(default_factory=dict)
    problems: list[str] = dc.field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """
            True if the graph describes a model the framework can instantiate.
        """
        return not self.problems

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def is_module_node(node: NodeModel) -> bool:
    """
        True if a node becomes a module of the controller.

        Sources, sinks and controller properties are not modules: they stand for the controller itself.
    """
    return not isinstance(node, (SourceNodeModel, SinkNodeModel, SelfPropertyNodeModel))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _port_map_for(port: PortModel) -> PortMap | None:
    """
        PortMap describing where the value arriving on an input port comes from.
    """
    node = port.node
    if node is None:
        return None
    if isinstance(node, SourceNodeModel):
        # A source stands for one input of the controller, named after the node.
        return PortMap(origin=_CALL_ORIGIN, port=node.name, is_property=False)
    if isinstance(node, SelfPropertyNodeModel):
        return PortMap(origin=_SELF_ORIGIN, port=node.name, is_property=True)
    is_property = port.compartment is not None and port.compartment is getattr(node, 'props_section', None)
    return PortMap(origin=node.name, port=port.name, is_property=is_property)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _incoming(port: PortModel) -> list[PortMap]:
    """
        Every connection feeding an input port, in a stable order.

        NOTE: The order matters. Several values arriving on the same port are concatenated by the controller
        in exactly this order.
    """
    maps = []
    for edge in port.edges:
        source = edge.source_port
        if source is None or source is port:
            continue
        port_map = _port_map_for(source)
        if port_map is not None:
            maps.append(port_map)
    return maps

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def build_module_specs(graph_model: GraphModel) -> tuple[list[ModuleSpecs], list[str]]:
    """
        Builds the ModuleSpecs of every module on the canvas.

        Returns:
            tuple, (specs, problems). Problems describe what would stop the model from being instantiated.
    """
    problems: list[str] = []
    specs: list[ModuleSpecs] = []

    names = [node.name for node in graph_model.nodes]
    duplicated = {name for name in names if names.count(name) > 1}
    for name in sorted(duplicated):
        problems.append(f'Several nodes are named "{name}". Every name must be unique.')

    # Outputs of the controller: a sink names one output and reads it from the module it is attached to.
    outputs_by_module: dict[str, dict[str, str]] = {}
    for node in graph_model.nodes:
        if not isinstance(node, SinkNodeModel):
            continue
        sources = _incoming(node.value_port)
        if not sources:
            problems.append(f'Output "{node.name}" is not connected to anything.')
            continue
        if len(sources) > 1:
            problems.append(f'Output "{node.name}" reads from several ports, only one is allowed.')
        origin = sources[0]
        if origin.origin in (_CALL_ORIGIN, _SELF_ORIGIN):
            problems.append(f'Output "{node.name}" must read from a module.')
            continue
        outputs_by_module.setdefault(origin.origin, {})[node.name] = origin.port

    for node in graph_model.nodes:
        if not is_module_node(node):
            continue
        module_cls = getattr(type(node), '_cls', None)
        if module_cls is None:
            problems.append(f'Node "{node.name}" is not backed by a Spark module and cannot be exported.')
            continue
        inputs: dict[str, list[PortMap]] = {}
        effects: dict[str, list[PortMap]] = {}
        for port in node.call_section.ports:
            if not port.is_input:
                continue
            maps = _incoming(port)
            if maps:
                inputs[port.name] = maps
            else:
                problems.append(f'Input "{port.name}" of "{node.name}" is not connected.')
        for port in node.props_section.ports:
            # A connection into a property is an effect: the module writes the value it receives.
            if not port.is_input:
                continue
            maps = _incoming(port)
            if maps:
                effects[port.name] = maps
        try:
            specs.append(ModuleSpecs(
                name=node.name,
                module_cls=module_cls,
                inputs=inputs,
                config=copy.deepcopy(node.config),
                outputs=outputs_by_module.get(node.name, {}),
                effects=effects,
            ))
        except Exception as error:
            problems.append(f'Unable to describe "{node.name}": {error}')
    return (specs, problems)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _unset_required_fields(config: SparkConfig, where: str) -> list[str]:
    """
        Fields left unset that the framework will need.

        NOTE: The inspector does not complain about these while editing, a model under construction is
        expected to be incomplete. They are reported here, where the graph is meant to be finished.
    """
    from spark.core.config import SparkConfig as _SparkConfig
    problems: list[str] = []
    if config is None:
        return [f'{where} has no configuration.']
    for field in dc.fields(config):
        value = getattr(config, field.name, None)
        if isinstance(value, _SparkConfig):
            problems.extend(_unset_required_fields(value, f'{where}.{field.name}'))
            continue
        if value is not None:
            continue
        if is_optional(type_tokens(field.type, field.metadata.get('valid_types'))):
            continue
        problems.append(f'"{where}.{field.name}" is not set.')
    return problems

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def build_controller_config(
        graph_model: GraphModel,
        strict: bool = False,
    ) -> ExportedGraph:
    """
        Translates a graph into the configuration of its controller.

        Input:
            graph_model: GraphModel, the graph to translate.
            strict: bool, also report everything that would stop the framework from instantiating the model.
                A session tolerates the problems, a model does not.

        Returns:
            ExportedGraph, the configuration, the specs, the node layout and the problems found.
    """
    profile: ControllerProfile | None = graph_model.profile
    result = ExportedGraph()
    if profile is None:
        result.problems.append('The graph has no controller, there is nothing to export.')
        return result

    specs, problems = build_module_specs(graph_model)
    result.specs = specs
    result.layout = {node.name: (float(node.pos[0]), float(node.pos[1])) for node in graph_model.nodes}
    if strict:
        result.problems.extend(problems)
    if not specs:
        result.problems.append('The graph does not contain any module.')

    config_cls = profile.config_cls
    if config_cls is None:
        result.problems.append(f'The configuration class of "{profile.label}" is not available.')
        return result
    try:
        # The controller keeps its own settings on the graph; only the modules come from the canvas.
        base = graph_model.controller_config
        own_fields = {}
        if base is not None:
            own_fields = {
                field.name: getattr(base, field.name, None)
                for field in dc.fields(base) if field.name != 'modules_specs'
            }
        # NOTE: partial() is what allows a half finished graph to be described at all: every field the user
        # has not set yet stays None instead of raising.
        result.config = config_cls.partial(modules_specs=tuple(specs), **own_fields)
    except Exception as error:
        result.problems.append(f'Unable to build the {profile.label} configuration: {error}')
        return result

    if strict:
        # The framework itself is the authority on whether the wiring is valid.
        controller_cls = profile.controller_cls
        if controller_cls is not None and specs:
            try:
                controller_cls._validate_modules(tuple(specs))
            except Exception as error:
                result.problems.append(str(error))
            else:
                try:
                    controller_cls._execution_order(tuple(specs))
                except Exception as error:
                    result.problems.append(str(error))
        for spec in specs:
            result.problems.extend(_unset_required_fields(spec.config, spec.name))
        result.problems.extend(_unset_required_fields(result.config, profile.label.lower()))
    return result

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
