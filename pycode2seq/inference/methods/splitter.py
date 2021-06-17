from abc import ABC, abstractmethod
from typing import List, Optional

from pycode2seq.inference.common.utils import pre_order
from pycode2seq.inference.common.node import Node
from pycode2seq.inference.methods.model import ElementNode, MethodInfo, MethodNode, ParameterNode
from pycode2seq.inference.parsing.utils import decompress_type_label


class Splitter(ABC):
    @abstractmethod
    def _method_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _method_return_type_node(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _method_name_node(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _class_declaration_node(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def _class_name_node(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _method_parameter_node(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _method_single_parameter_node(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def _parameter_return_type_node(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _parameter_name_node(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _get_list_of_parameters(self, root: Node) -> List[ParameterNode]:
        raise NotImplementedError

    def split_on_methods(self, root: Node) -> List[MethodInfo]:
        method_roots = [node for node in pre_order(root) if
                        decompress_type_label(node.type_label)[-1] == self._method_name()]
        return [self._collect_method_info(root) for root in method_roots]

    def _collect_method_info(self, method_node: Node) -> MethodInfo:
        method_name = method_node.get_child_of_type(self._method_name_node())
        method_return_type_node = method_node.get_child_of_type(self._method_return_type_node())
        if method_return_type_node:
            method_return_type_node.token = self._collect_parameter_token(method_return_type_node)

        class_root = self._get_enclosing_class(method_node)
        class_name = class_root.get_child_of_type(self._class_name_node()) if class_root is not None else None

        parameters_root = method_node.get_child_of_type(self._method_parameter_node())
        parameters_list = self._get_list_of_parameters(parameters_root)

        return MethodInfo(
            MethodNode(method_node, method_return_type_node, method_name),
            ElementNode(class_root, class_name),
            parameters_list
        )

    def _get_enclosing_class(self, root: Node) -> Optional[Node]:
        if decompress_type_label(root.type_label)[-1] in self._class_declaration_node():
            return root

        parent = root.parent
        if parent is not None:
            return self._get_enclosing_class(parent)

        return None

    def _get_parameter_info_from_node(self, root: Node) -> ParameterNode:
        return_type_node = root.get_child_of_type(self._parameter_return_type_node())
        return_type_node.token = self._collect_parameter_token(return_type_node)

        return ParameterNode(
            root,
            return_type_node,
            root.get_child_of_type(self._parameter_name_node())
        )

    def _collect_parameter_token(self, root: Node) -> str:
        if root.is_leaf():
            return root.token

        return "".join([self._collect_parameter_token(child) for child in root.children])
