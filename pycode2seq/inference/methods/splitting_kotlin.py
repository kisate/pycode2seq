from typing import List

from pycode2seq.inference.common.node import Node
from pycode2seq.inference.methods.model import ParameterNode
from pycode2seq.inference.methods.splitter import Splitter
from pycode2seq.inference.parsing.utils import decompress_type_label


class KotlinSplitter(Splitter):

    def _method_name(self) -> str:
        return "functionDeclaration"

    def _method_return_type_node(self) -> str:
        return "type"

    def _method_name_node(self) -> str:
        return "simpleIdentifier"

    def _class_declaration_node(self) -> List[str]:
        return ["classDeclaration", "objectDeclaration"]

    def _class_name_node(self) -> str:
        return "simpleIdentifier"

    def _method_parameter_node(self) -> str:
        return "functionValueParameters"

    def _method_single_parameter_node(self) -> List[str]:
        return ["functionValueParameter"]

    def _parameter_return_type_node(self) -> str:
        return "type"

    def _parameter_name_node(self) -> str:
        return "simpleIdentifier"

    def _get_list_of_parameters(self, root: Node) -> List[ParameterNode]:
        if root is None:
            return []
        return [
            self._get_parameter_info_from_node(child) for child in root.children
            if decompress_type_label(child.type_label)[0] == self._method_single_parameter_node()[0]
        ]
