from typing import List, Optional
from antlr4.tree.Tree import TerminalNode
from pycode2seq.inference.common.node import Node
from antlr4 import ParserRuleContext
from antlr4 import Lexer


def convert_rule_context(cntx: ParserRuleContext, rule_names: List[str], parent: Optional[Node], lexer: Lexer) -> Node:
    type_label = rule_names[cntx.getRuleIndex()]
    current_node = Node(type_label, parent, None)
    children = []
    for child in cntx.getChildren():
        if isinstance(child, TerminalNode):
            children.append(convert_terminal(child, current_node, lexer))
        else:
            children.append(convert_rule_context(child, rule_names, current_node, lexer))

    current_node.set_children(children)

    return current_node


def convert_terminal(terminal_node: TerminalNode, parent: Node, lexer: Lexer):
    return Node(lexer.symbolicNames[terminal_node.getSymbol().type], parent, terminal_node.getSymbol().text)


def compress_tree(root: Node) -> Node:
    if len(root.children) == 1:
        child = compress_tree(root.children[0])
        compressed_node = Node(
            root.type_label + "|" + child.type_label,
            root.parent,
            child.get_token()
        )
        compressed_node.set_children(child.children)
        return compressed_node

    root.set_children([compress_tree(child) for child in root.children])
    return root


def decompress_type_label(type_label: str) -> List[str]:
    return type_label.split("|")


def shorten_nodes(root: Node) -> Node:
    parts = decompress_type_label(root.type_label)
    label = parts[0]

    if len(parts) > 1:
        label += "/" + parts[-1]

    children = [shorten_nodes(child) for child in root.children]

    new_node = Node(label, root.parent, root.token)

    new_node.set_children(children)

    return new_node
