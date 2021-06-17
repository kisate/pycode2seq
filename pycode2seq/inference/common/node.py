from typing import List


class Node:

    def __init__(self, type_label: str, parent: "Node", token: str) -> None:
        self.type_label = type_label
        self.parent = parent
        self.token = token
        self.metadata = {}
        self.children: List["Node"] = []

    def get_token(self) -> str:
        if self.token is None:
            return "None"
        return self.token

    def get_normalized_token(self) -> str:
        from pycode2seq.inference.common.utils import normalize_token
        if self.token is None:
            return ""
        return normalize_token(self.token, "")

    def pretty_print(self, indent=0, indent_symbol="--") -> None:
        print(indent_symbol * indent, end="")
        print(self.type_label, end="")

        if self.get_token():
            print(f" : {self.get_token()}")
        else:
            print()

        for child in self.children:
            child.pretty_print(indent + 1, indent_symbol)

    def set_children(self, new_children) -> None:
        self.children = new_children
        for child in self.children:
            child.parent = self

    def get_children_of_type(self, type_label: str) -> List["Node"]:
        from pycode2seq.inference.parsing.utils import decompress_type_label
        return [child for child in self.children if
                next(iter(decompress_type_label(child.type_label)), None) == type_label]

    def get_child_of_type(self, type_label: str) -> "Node":
        return next(iter(self.get_children_of_type(type_label)), None)

    def is_leaf(self):
        return len(self.children) == 0
