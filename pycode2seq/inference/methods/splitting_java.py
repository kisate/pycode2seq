from typing import List

from pycode2seq.inference.common.node import Node
from pycode2seq.inference.methods.model import ParameterNode
from pycode2seq.inference.methods.splitter import Splitter
from pycode2seq.inference.parsing.utils import decompress_type_label


class JavaSplitter(Splitter):

    def _method_name(self) -> str:
        return "methodDeclaration"

    def _method_return_type_node(self) -> str:
        return "typeTypeOrVoid"

    def _method_name_node(self) -> str:
        return "IDENTIFIER"

    def _class_declaration_node(self) -> List[str]:
        return ["classDeclaration"]

    def _class_name_node(self) -> str:
        return "IDENTIFIER"

    def _method_parameter_node(self) -> str:
        return "formalParameters"

    def _method_single_parameter_node(self) -> List[str]:
        return ["formalParameter", "lastFormalParameter"]

    def _parameter_return_type_node(self) -> str:
        return "typeType"

    def _parameter_name_node(self) -> str:
        return "variableDeclaratorId"

    def _get_list_of_parameters(self, root: Node) -> List[ParameterNode]:
        if root is None:
            return []

        inner_parameters_root = root.get_child_of_type("formalParameterList")
        params_root = root if inner_parameters_root is None else inner_parameters_root

        if decompress_type_label(params_root.type_label)[-1] == self._method_single_parameter_node():
            return [self._get_parameter_info_from_node(params_root)]

        return [
            self._get_parameter_info_from_node(child) for child in params_root.children
            if decompress_type_label(child.type_label)[0] in self._method_single_parameter_node()
        ]
