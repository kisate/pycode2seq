from typing import List

from pycode2seq.inference.common.node import Node
from pycode2seq.inference.methods.model import MethodInfo
from pycode2seq.inference.methods.splitter import Splitter
from pycode2seq.inference.methods.splitting_java import JavaSplitter
from pycode2seq.inference.methods.splitting_kotlin import KotlinSplitter
from pycode2seq.inference.parsing.languages import JavaParser, KotlinParser
from pycode2seq.inference.parsing.parser import Parser


class Language:
    def __init__(self, parser: Parser, splitter: Splitter):
        self.parser = parser
        self.splitter = splitter

    def parse(self, file_path: str) -> Node:
        return self.parser.parse(file_path)

    def split_on_methods(self, root: Node) -> List[MethodInfo]:
        return self.splitter.split_on_methods(root)

    @staticmethod
    def by_name(name: str) -> 'Language':
        if name == "java":
            return java
        elif name == "kt" or name == "kotlin":
            return kotlin
        else:
            raise ValueError(f"Unknown language {name}")


java = Language(JavaParser(), JavaSplitter())
kotlin = Language(KotlinParser(), KotlinSplitter())
