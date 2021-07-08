from typing import Dict, List, Optional
from antlr4.tree.Tree import TerminalNode
from pycode2seq.inference.common.node import Node
from antlr4 import ParserRuleContext
from antlr4 import Lexer

import numpy as np

from os import path

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

def _get_id2value_from_csv(path_: str) -> Dict[str, str]:
    return dict(np.genfromtxt(path_, delimiter=",", dtype=(str, str))[1:])


def read_astminer(data_path: str) -> List[str]:
    id_to_token_data_path = path.join(data_path, f"tokens.csv")
    id_to_type_data_path = path.join(data_path, f"node_types.csv")
    id_to_paths_data_path = path.join(data_path, f"paths.csv")
    path_contexts_path = path.join(data_path, f"path_contexts.csv")

    id_to_paths_stored = _get_id2value_from_csv(id_to_paths_data_path)
    id_to_paths = {index: [n for n in nodes.split()] for index, nodes in id_to_paths_stored.items()}

    id_to_node_types = _get_id2value_from_csv(id_to_type_data_path)
    id_to_node_types = {index: node_type.rsplit(" ", maxsplit=1)[0] for index, node_type in id_to_node_types.items()}

    id_to_tokens = _get_id2value_from_csv(id_to_token_data_path)
    

    with open(path_contexts_path, "r") as path_contexts_file:
        output_lines = []
        for line in path_contexts_file:
            label, *path_contexts = line.split()
            parsed_line = [label]
            for path_context in path_contexts:
                from_token_id, path_types_id, to_token_id = path_context.split(",")
                from_token, to_token = id_to_tokens[from_token_id], id_to_tokens[to_token_id]
                nodes = [id_to_node_types[p_] for p_ in id_to_paths[path_types_id]]
                parsed_line.append(",".join([from_token, "|".join(nodes), to_token]))
            output_lines.append(" ".join(parsed_line))
        
        return output_lines