from typing import List
from dataclasses import dataclass

from pycode2seq.inference.common.node import Node


@dataclass
class MethodNode:
    root: Node
    return_type_node: Node
    name_node: Node

    @property
    def name(self):
        return self.name_node.get_token()

    @property
    def return_type(self):
        return self.return_type_node.get_token()


@dataclass
class ElementNode:
    root: Node
    name_node: Node

    @property
    def name(self):
        return self.name_node.get_token()


@dataclass
class ParameterNode:
    root: Node
    return_type_node: Node
    name_node: Node

    @property
    def name(self):
        return self.name_node.get_token()

    @property
    def return_type(self):
        return self.return_type_node.get_token()


@dataclass
class MethodInfo:
    method: MethodNode
    enclosing_element: ElementNode
    method_parameters: List[ParameterNode]

    @property
    def name(self):
        return self.method.name

    @property
    def return_type(self):
        return self.method.return_type

    @property
    def enclosing_element_name(self):
        return self.enclosing_element.name
