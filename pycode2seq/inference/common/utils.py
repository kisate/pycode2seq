import regex as re

from typing import List

from pycode2seq.inference.common.node import Node


TECHNICAL_TOKEN_KEY = "technical_token"
DEFAULT_TOKEN = "EMPTY_TOKEN"


def do_traverse_pre_order(node: Node, result: List[Node]):
    result.append(node)
    for child in node.children:
        do_traverse_pre_order(child, result)


def pre_order(root: Node) -> List[Node]:
    result = []
    do_traverse_pre_order(root, result)
    return result


def set_node_technical_token(node: Node, token: str):
    node.metadata[TECHNICAL_TOKEN_KEY] = token


def normalize_token(token: str, default: str) -> str:
    clean_token = re.sub("\\P{Print}", "", re.sub("[\"',]", "", re.sub("//s+", "", re.sub("\\\\n", "", token.lower()))))

    stripped = re.sub("[^A-Za-z]", "", clean_token)

    if not stripped:
        careful_stripped = clean_token.replace(" ", "_")
        if not careful_stripped:
            return default
        return careful_stripped

    return stripped


def split_to_subtokens(token: str) -> List[str]:
    splitted = re.split("(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+", token.strip())
    return [normalize_token(token, "") for token in splitted if token]
