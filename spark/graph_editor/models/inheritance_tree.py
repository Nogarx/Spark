#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations

import enum
import typing as tp
import copy 
import dataclasses as dc
from spark.core.utils import ascii_tree

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

    def __post_init__(self,) -> None:
        if isinstance(self.type_string, tp.Iterable):
            type_string = set(self.type_string)
            self.type_string = [t.__name__ if isinstance(t, type) else str(t) for t in type_string if t is not None]
        elif isinstance(self.type_string, type):
            self.type_string = self.type_string.__name__

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
            Cheks the leaf node can inherit.
        """
        return bool(self.flags & InheritanceFlags.CAN_INHERIT)
    
    def is_inheriting(self,) -> bool:
        """
            Cheks the leaf node is inheriting.
        """
        return bool(self.flags & InheritanceFlags.IS_INHERITING)

    def can_receive(self,) -> bool:
        """
            Cheks the leaf node can receive.
        """
        return bool(self.flags & InheritanceFlags.CAN_RECEIVE)
    
    def is_receiving(self,) -> bool:
        """
            Cheks the leaf node is receiving.
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
        Tree-like data structure to manage the inheritance status of variables in the Spark Graph Editor. 

        This data structure is used to link variables with the same names and types for simultaneous updates within the GUI. 
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
            Parses the tree with to produce a string with the appropiate format for the ascii_tree method.
        """

        rep = current_depth * ' ' + f'{self._current_path[-1]}\n' if len(self._current_path) > 0 else ''
        for l, s in self._leaves.items():
            rep += (current_depth + 1) * ' ' + f'{l}: {s.flags}\n'
        for _, t in self._branches.items():
            rep += t._parse_tree_with_spaces(current_depth+1 if len(self._current_path) > 0 else 0)
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
        # Make a copy of the path to prevent overrides
        path = copy.deepcopy(path if isinstance(path, list) else list(path))
        if len(path) == 1:
            # Add leave to current level.
            self._leaves[path[0]] = InheritanceLeaf(
                name=path[0], 
                type_string=type_string,
                inheritance_childs=inheritance_childs,
                flags=flags, 
                break_inheritance=break_inheritance,
                parent=self,
            )
        elif len(path) > 1:
            # Consume leading path string.
            branch = path.pop(0)
            # Allow adding branches to simplify usage.
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
        # Make a copy of the path to prevent overrides
        path = copy.deepcopy(path if isinstance(path, list) else list(path))
        if len(path) == 1:
            # Add branch to current level.
            self._branches[path[0]] = InheritanceTree(self._current_path + path)
        elif len(path) > 1:
            # Consume leading path string.
            branch = path.pop(0)
            # Allow adding branches recursively to simplify usage.
            if branch not in self._branches:
                self._branches[branch] = InheritanceTree(self._current_path + branch)
            self._branches[branch].add_branch(path)
        else: 
            raise ValueError(
                f'Invalid path, got: {path}. Path must point to final branch.'
            )
        
    def validate(self, inheriting_labels: dict = {}) -> None:
        """
            Validates the flags and the inheritance childs of the tree.
        """
        inheriting_labels = copy.deepcopy(inheriting_labels)
        if self._is_valid:
            return
        if len(self._branches) == 0:
            # If there are no branches, leaves cannot inherit.
            can_inherit = InheritanceFlags(0)
            is_inheriting = InheritanceFlags(0)
            for l in self._leaves.keys():
                # If label is not in inheriting_labels, leaf cannot receive.
                if l not in inheriting_labels:
                    can_receive = InheritanceFlags(0)
                    is_receiving = InheritanceFlags(0)
                else:
                    can_receive = InheritanceFlags.CAN_RECEIVE
                    is_receiving = InheritanceFlags.IS_RECEIVING if inheriting_labels[l] else InheritanceFlags(0)
                # Set leaf attributes
                if self._leaves[l].break_inheritance:
                    self._leaves[l].flags = InheritanceFlags(0)
                else:
                    self._leaves[l].flags = can_inherit | is_inheriting | can_receive | is_receiving
                self._leaves[l].inheritance_childs = []
        else:
            for l, il in self._leaves.items():
                # Get leaf inheritance_childs
                inheritance_childs = self._compute_leaf_childs(l)
                # Check if can inherit its value
                can_inherit = InheritanceFlags.CAN_INHERIT if len(inheritance_childs) > 0 else InheritanceFlags(0)
                # Preseve is_inheriting flag unless it is set on by error.
                is_inheriting = il.flags & InheritanceFlags.IS_INHERITING if can_inherit else InheritanceFlags(0)
                # If label is not in inheriting_labels, leaf cannot receive.
                if l not in inheriting_labels:
                    can_receive = InheritanceFlags(0)
                    is_receiving = InheritanceFlags(0)
                else:
                    can_receive = InheritanceFlags.CAN_RECEIVE
                    is_receiving = InheritanceFlags.IS_RECEIVING if inheriting_labels[l] else InheritanceFlags(0)
                # Set leaf attributes
                if self._leaves[l].break_inheritance:
                    self._leaves[l].flags = InheritanceFlags(0)
                else:
                    self._leaves[l].flags = can_inherit | is_inheriting | can_receive | is_receiving
                self._leaves[l].inheritance_childs = inheritance_childs

                # Add leaf to inheriting labels
                if len(inheritance_childs) > 0:
                    inheriting_labels[l] = inheriting_labels.get(l, None) or bool(is_inheriting)

        # Validate branches:
        for b in self._branches.keys():
            self._branches[b].validate(inheriting_labels)
        # Flag
        self._is_valid = True

    # NOTE: This method should be computed from deeper branches to shallow for efficiency. However, in practice Inheritance
    # trees will not have more than a few levels and a couple dozens of parameters which makes forward search acceptable.
    def _compute_leaf_childs(self, name: str, path: list[str] = []) -> list[list[str]]:
        """
            Collects the inheritance childs of a tree, relative to the current leaf.

            Input:
                name: str, leaf node name to search

            Returns:
                list[list[str]], list of inheritance childs of the leaf node
        """
        inheritance_childs = []
        # Search in subtrees only.
        if len(path) > 0:
            for l, lo in self._leaves.items():
                if l == name:
                    # Check if leaf has the break_inheritance flag
                    if not lo.break_inheritance:
                        inheritance_childs.append(path + [name])
                    break
        for b in self._branches.keys():
            inheritance_childs += self._branches[b]._compute_leaf_childs(name, [b])
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
        # Make a copy of the path to prevent overrides
        path = copy.deepcopy(path if isinstance(path, list) else list(path))
        if len(path) == 1:
            # Add branch to current level.
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
        # Make a copy of the path to prevent overrides
        path = copy.deepcopy(path if isinstance(path, list) else list(path))
        if len(path) == 1:
            # Add branch to current level.
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
        # Collect leaves
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