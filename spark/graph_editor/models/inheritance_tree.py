#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import enum
import typing as tp
import copy
import dataclasses as dc
from spark.core.utils import ascii_tree
from spark.graph_editor.models.config_types import type_tokens

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class InheritanceFlags(enum.IntFlag):
    CAN_INHERIT = 0b1000
    IS_INHERITING = 0b0100
    CAN_RECEIVE = 0b0010
    IS_RECEIVING = 0b0001

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@dc.dataclass
class InheritanceLeaf:
    """
        Leaf object for the InheritanceTree data structure.
    """

    name: str
    type_string: str
    inheritance_childs: list[list[str]]
    flags: InheritanceFlags = 0b0000
    break_inheritance: bool = False
    parent: InheritanceTree = None
    type_key: frozenset[str] = dc.field(init=False, default_factory=frozenset)

    def __post_init__(self,) -> None:
        # NOTE: A type_string may arrive as a plain annotation string, a type, or the "valid_types" metadata
        # tuple. Strings are iterable, so they are handled before the generic iterable branch.
        raw = self.type_string
        if raw is None:
            names = []
        elif isinstance(raw, str):
            names = [raw]
        elif isinstance(raw, type):
            names = [raw.__name__]
        elif isinstance(raw, tp.Iterable):
            names = [t.__name__ if isinstance(t, type) else str(t) for t in raw if t is not None]
        else:
            names = [str(raw)]
        self.type_string = names[0] if len(names) == 1 else names
        # Cascading is only allowed between fields that describe the same type.
        self.type_key = type_tokens(valid_types=tuple(names))
        # Normalize the flags, so raw integers are usable as InheritanceFlags.
        self.flags = InheritanceFlags(self.flags)

    def __repr__(self,) -> str:
        rep = f'{self.name}\n'
        rep += ' ' + f'type_string: {self.type_string}\n'
        rep += ' ' + f'flags: {self.flags}\n'
        rep += ' ' + f'break_inheritance: {self.break_inheritance}\n'
        rep += ' ' + f'inheritance_childs:\n'
        for c in self.inheritance_childs:
            rep += 2*' ' + f'{c}\n' 
        return ascii_tree(rep)


    def to_dict(self,) -> dict:
        return {
            'name': self.name,
            'type_string': self.type_string,
            'inheritance_childs': self.inheritance_childs,
            'break_inheritance': self.break_inheritance,
            'flags': self.flags.value
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'InheritanceLeaf':
        d = copy.deepcopy(d)
        d['flags'] = InheritanceFlags(d['flags'])
        return cls(**d)

    def can_inherit(self,) -> bool:
        """
            Checks the leaf node can inherit.
        """
        return bool(self.flags & InheritanceFlags.CAN_INHERIT)
    
    def is_inheriting(self,) -> bool:
        """
            Checks the leaf node is inheriting.
        """
        return bool(self.flags & InheritanceFlags.IS_INHERITING)

    def can_receive(self,) -> bool:
        """
            Checks the leaf node can receive.
        """
        return bool(self.flags & InheritanceFlags.CAN_RECEIVE)
    
    def is_receiving(self,) -> bool:
        """
            Checks the leaf node is receiving.
        """
        return bool(self.flags & InheritanceFlags.IS_RECEIVING)

    @property
    def path(self,) -> list[str]:
        """
            Returns the path of the leaf node.
        """
        return self.parent.path + [self.name]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class InheritanceTree:
    """
        Tree-like data structure holding the inheritance status of the variables of a node.

        Links variables of the same name and type, so that they are updated simultaneously.
        """

    def __init__(self, path: list[str] = []) -> None:
        self._is_valid = False
        self._current_path = path
        self._leaves: dict[str, InheritanceLeaf] = {}
        self._branches: dict[str, InheritanceTree] = {}

    def __repr__(self,) -> str:
        if not self._is_valid:
            self.validate()
        r = self._parse_tree_with_spaces(0)
        return ascii_tree(r)

    def _parse_tree_with_spaces(self, current_depth: int) -> str:
        """
            Parses the tree into the format expected by the ascii_tree method.
        """

        # NOTE: The root of a tree has an empty path, but ascii_tree requires a depth 0 header.
        name = self._current_path[-1] if len(self._current_path) > 0 else 'config'
        rep = current_depth * ' ' + f'{name}\n'
        for l, s in self._leaves.items():
            rep += (current_depth + 1) * ' ' + f'{l}: {s.flags}\n'
        for _, t in self._branches.items():
            rep += t._parse_tree_with_spaces(current_depth + 1)
        return rep

    def add_leaf(
            self, 
            path: list[str], 
            type_string: str = '', 
            inheritance_childs: list[list[str]]=[], 
            flags: InheritanceFlags = 0b0000,
            break_inheritance: bool = False,
            **kwargs,
        ) -> None:
        """
            Adds a new leaf to the tree.

            Input:
                path: list[str], path to the new leaf node, with the last entry the name of the leaf
                type_string: str, string representation of the types this variable manages
                inheritance_childs: list[list[str]]=[], list of children that can inherit from this variable (Note: do not set by hand)
                flags: InheritanceFlags, 4-bit flags that represent inheritance possibilities (Note: do not set by hand)
                break_inheritance: bool, boolean flag to disconnect this variable from the inheritance dynamics
        """
        # The path is consumed below, so the list of the caller is left untouched.
        path = copy.deepcopy(path if isinstance(path, list) else list(path))
        if len(path) == 1:
            self._leaves[path[0]] = InheritanceLeaf(
                name=path[0], 
                type_string=type_string,
                inheritance_childs=inheritance_childs,
                flags=flags, 
                break_inheritance=break_inheritance,
                parent=self,
            )
        elif len(path) > 1:
            branch = path.pop(0)
            # Adding a leaf creates the branches it hangs from.
            if branch not in self._branches:
                self.add_branch([branch])
            self._branches[branch].add_leaf(
                path, 
                type_string=type_string, 
                flags=flags, 
                break_inheritance=break_inheritance
            )
        else: 
            raise ValueError(
                f'Invalid path, got: {path}. Path must point to final leaf.'
            )
        self._is_valid = False
        
    def add_branch(self, path: list[str]) -> None:
        """
            Adds a new branch to the tree.

            Input:
                path: list[str], path to the new branch, with the last entry the name of the branch
        """
        # The path is consumed below, so the list of the caller is left untouched.
        path = copy.deepcopy(path if isinstance(path, list) else list(path))
        if len(path) == 1:
            self._branches[path[0]] = InheritanceTree(self._current_path + path)
        elif len(path) > 1:
            branch = path.pop(0)
            # Adding a branch creates the branches it hangs from.
            if branch not in self._branches:
                self._branches[branch] = InheritanceTree(self._current_path + [branch])
            self._branches[branch].add_branch(path)
        else: 
            raise ValueError(
                f'Invalid path, got: {path}. Path must point to final branch.'
            )
        
    def invalidate(self,) -> None:
        """
            Marks the whole subtree as invalid, forcing a full recomputation on the next validate() call.
        """
        self._is_valid = False
        for branch in self._branches.values():
            branch.invalidate()

    def validate(self, inheriting_labels: dict | None = None) -> None:
        """
            Validates the flags and the inheritance childs of the tree.

            Input:
                inheriting_labels: dict, {(leaf name, leaf type): is_inheriting} entries contributed by the
                    ancestors of this subtree. Leaves matching an entry are marked as receiving.
        """
        # NOTE: Labels are keyed by (name, type), so same-named fields of different types never cascade into
        # each other.
        inheriting_labels = dict(inheriting_labels) if inheriting_labels else {}
        if self._is_valid:
            return
        for name, leaf in self._leaves.items():
            key = (name, leaf.type_key)
            inheritance_childs = self._compute_leaf_childs(name, leaf.type_key)
            can_inherit = InheritanceFlags.CAN_INHERIT if len(inheritance_childs) > 0 else InheritanceFlags(0)
            # Preserve the is_inheriting flag unless it was set on a leaf that cannot inherit.
            is_inheriting = leaf.flags & InheritanceFlags.IS_INHERITING if can_inherit else InheritanceFlags(0)
            if key not in inheriting_labels:
                can_receive = InheritanceFlags(0)
                is_receiving = InheritanceFlags(0)
            else:
                can_receive = InheritanceFlags.CAN_RECEIVE
                is_receiving = InheritanceFlags.IS_RECEIVING if inheriting_labels[key] else InheritanceFlags(0)
            if leaf.break_inheritance:
                leaf.flags = InheritanceFlags(0)
                leaf.inheritance_childs = []
                continue
            leaf.flags = can_inherit | is_inheriting | can_receive | is_receiving
            leaf.inheritance_childs = inheritance_childs
            if len(inheritance_childs) > 0:
                inheriting_labels[key] = inheriting_labels.get(key, False) or bool(is_inheriting)

        for b in self._branches.keys():
            self._branches[b].validate(inheriting_labels)
        self._is_valid = True

    # NOTE: Searching from the deeper branches to the shallow ones would be cheaper. Inheritance trees hold a
    # few levels and a couple dozen parameters, which makes the forward search acceptable.
    def _compute_leaf_childs(self, name: str, type_key: frozenset[str], path: list[str] | None = None) -> list[list[str]]:
        """
            Collects the inheritance childs of a tree, relative to the current leaf.

            Input:
                name: str, leaf node name to search
                type_key: frozenset[str], normalized type of the leaf node. Only childs describing the same
                    type are considered valid cascade targets.

            Returns:
                list[list[str]], list of inheritance childs of the leaf node
        """
        path = [] if path is None else path
        inheritance_childs = []
        if len(path) > 0:
            for l, lo in self._leaves.items():
                if l == name and lo.type_key == type_key:
                    if not lo.break_inheritance:
                        inheritance_childs.append(path + [name])
                    break
        for b in self._branches.keys():
            inheritance_childs += self._branches[b]._compute_leaf_childs(name, type_key, path + [b])
        return inheritance_childs

    def get_leaf(self, path: list[str]) -> InheritanceLeaf:
        """
            Returns the status of the leaf node.

            Input:
                path: list[str], path to the leaf node, with the last entry the name of the leaf

            Returns:
                InheritanceLeaf, returns the leaf node instance.
        """
        if not self._is_valid:
            self.validate()
        # The path is consumed below, so the list of the caller is left untouched.
        path = copy.deepcopy(path if isinstance(path, list) else list(path))
        if len(path) == 1:
            node = self._leaves.get(path[0], None)
            if node:
                return node
            else:
                raise KeyError(
                    f'Node \"{path}\" not found.'
                )
        elif len(path) > 1:
            branch = path.pop(0)
            subtree = self._branches.get(branch, None)
            if subtree is None:
                raise KeyError(
                    f'Subtree \"{path}\" not found.'
                )
            else:
                return subtree.get_leaf(path)
    
    def get_subtree(self, path: list[str]) -> InheritanceTree:
        """
            Returns a subtree of the leaf node.

            Input:
                path: list[str], path to the subtree node, with the last entry the name of the branch

            Returns:
                InheritanceTree, returns the branch node instance.
        """
        if not self._is_valid:
            self.validate()
        # The path is consumed below, so the list of the caller is left untouched.
        path = copy.deepcopy(path if isinstance(path, list) else list(path))
        if len(path) == 1:
            subtree = self._branches.get(path[0], None)
            if subtree is None:
                raise KeyError(
                    f'Subtree \"{path}\" not found.'
                )
        elif len(path) > 1:
            branch = path.pop(0)
            subtree = self._branches.get(branch, None)
            if subtree is None:
                raise KeyError(
                    f'Subtree \"{path}\" not found.'
                )
            return subtree.get_subtree(path) if subtree else None
        
    def to_dict(self,) -> dict:
        """
            InheritanceTree dict serializer.
        """
        if not self._is_valid:
            self.validate()
        return {
            **{l: il.to_dict() for l, il in self._leaves.items()},
            **{b: t.to_dict() for b, t in self._branches.items()}
        }
    
    @classmethod
    def from_dict(cls, d: dict, path: list[str] = []) -> 'InheritanceTree':
        """
            InheritanceTree dict deserializer.
        """
        tree = cls(path)
        for k, v in d.items():
            if isinstance(v, dict):
                if v.get('flags', None) is not None:
                    tree.add_leaf([k], **v)
                else:
                    tree.add_branch([k])
                    tree._branches[k] = cls.from_dict(v, path=path+[k])
            else:
                raise TypeError(
                    f'Expected \"v\" to be a dict, but got \"{v}\".'
                )
        return tree

    @property
    def path(self,) -> list[str]:
        """
            Returns the path of the branch node.
        """
        return self._current_path

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################