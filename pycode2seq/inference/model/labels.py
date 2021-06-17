from dataclasses import dataclass
from typing import List, Callable

from pycode2seq.inference.methods.extracting import extract_label
from pycode2seq.inference.common.node import Node
from pycode2seq.inference.methods.model import MethodInfo
from pycode2seq.inference.paths.extracting import ASTPath, ExtractingParams, extract_paths


@dataclass
class LabeledData:
    label: str
    method_name: str
    paths: List[ASTPath]

    def __str__(self) -> str:
        return self.label + " " + " ".join(str(path) for path in self.paths)


def extract_labels_with_paths(root: Node, params: ExtractingParams,
                              split_into_methods: Callable[[Node], List[MethodInfo]]) -> List[LabeledData]:
    methods = split_into_methods(root)
    return [LabeledData(extract_label(method, True), method.name, extract_paths(method.method.root, params)) for method in methods]
