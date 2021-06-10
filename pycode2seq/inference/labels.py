from typing import List
from pycode2seq.inference.methods.extracting import extract_label
from pycode2seq.inference.common.node import Node
from pycode2seq.inference.paths.extracting import ASTPath, ExtractingParams, extract_paths
from dataclasses import dataclass

@dataclass
class LabeledData:
    label: str
    paths: List[ASTPath]

    def __str__(self) -> str:
        return self.label + " " + " ".join(str(path) for path in self.paths)

def extract_labels_with_paths(root: Node, params: ExtractingParams, split_into_methods) -> List[LabeledData]:
    methods = split_into_methods(root)
    return [LabeledData(extract_label(method, True), extract_paths(method.method.root, params)) for method in methods]