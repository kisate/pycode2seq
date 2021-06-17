from abc import ABC, abstractmethod
from typing import List, Optional, Callable

from antlr4 import FileStream, CommonTokenStream, Lexer, InputStream, Parser as AntlrParser, \
    TokenStream, ParserRuleContext, TerminalNode

from pycode2seq.inference.common.node import Node


class Parser(ABC):
    @abstractmethod
    def _lexer(self, input_stream: InputStream) -> Lexer:
        raise NotImplementedError

    @abstractmethod
    def _parser(self, stream: TokenStream) -> AntlrParser:
        raise NotImplementedError

    @abstractmethod
    def _get_tree(self, parser: AntlrParser) -> ParserRuleContext:
        raise NotImplementedError

    def parse(self, file_path: str) -> Node:
        input_stream = FileStream(file_path)
        lexer = self._lexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = self._parser(stream)
        tree = self._get_tree(parser)

        return Parser._compress_tree(Parser._convert_rule_context(tree, parser.ruleNames, None, lexer))

    @staticmethod
    def _convert_rule_context(cntx: ParserRuleContext, rule_names: List[str], parent: Optional[Node],
                              lexer: Lexer) -> Node:
        type_label = rule_names[cntx.getRuleIndex()]
        current_node = Node(type_label, parent, None)
        children = []
        for child in cntx.getChildren():
            if isinstance(child, TerminalNode):
                children.append(Parser._convert_terminal(child, current_node, lexer))
            else:
                children.append(Parser._convert_rule_context(child, rule_names, current_node, lexer))

        current_node.set_children(children)

        return current_node

    @staticmethod
    def _convert_terminal(terminal_node: TerminalNode, parent: Node, lexer: Lexer):
        return Node(lexer.symbolicNames[terminal_node.getSymbol().type], parent, terminal_node.getSymbol().text)

    @staticmethod
    def _compress_tree(root: Node) -> Node:
        if len(root.children) == 1:
            child = Parser._compress_tree(root.children[0])
            compressed_node = Node(
                root.type_label + "|" + child.type_label,
                root.parent,
                child.get_token()
            )
            compressed_node.set_children(child.children)
            return compressed_node

        root.set_children([Parser._compress_tree(child) for child in root.children])
        return root

    @staticmethod
    def _decompress_type_label(type_label: str) -> List[str]:
        return type_label.split("|")

    @staticmethod
    def _shorten_nodes(root: Node) -> Node:
        parts = Parser._decompress_type_label(root.type_label)
        label = parts[0]

        if len(parts) > 1:
            label += "/" + parts[-1]

        children = [Parser._shorten_nodes(child) for child in root.children]

        new_node = Node(label, root.parent, root.token)

        new_node.set_children(children)

        return new_node
