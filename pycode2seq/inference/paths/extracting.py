from dataclasses import dataclass
from typing import List
from pycode2seq.inference.common.node import Node


@dataclass
class PathPiece:
    nodes: List[Node]
    terminal: Node

    def __str__(self) -> str:
        return "->".join(node.type_label for node in self.nodes) + f"->({self.terminal.get_token()})"

    def __len__(self) -> int:
        return len(self.nodes)


@dataclass
class ASTPath:
    nodes: List[Node]
    left_terminal: Node
    right_terminal: Node

    def __str__(self) -> str:
        return f"{self.left_terminal.get_normalized_token()}," + "|".join(
            node.type_label for node in self.nodes) + f",{self.right_terminal.get_normalized_token()}"


@dataclass
class ExtractingParams:
    max_length: int
    max_width: int
    paths_per_method: int


def extract_paths(root: Node, params: ExtractingParams) -> List[ASTPath]:
    paths = []
    extract_pieces(root, paths, params)
    return paths


def collect_paths(pieces: List[List[PathPiece]], node: Node, params: ExtractingParams) -> List[ASTPath]:
    paths = []
    for i, left_pieces in enumerate(pieces):
        for right_pieces in pieces[i + 1: i + 1 + params.max_width]:
            for left_piece in left_pieces:
                for right_piece in right_pieces:
                    left_token = left_piece.terminal.get_normalized_token()
                    right_token = right_piece.terminal.get_normalized_token()
                    if len(left_piece) + len(right_piece) + 1 <= params.max_length and left_token and right_token:
                        path = left_piece.nodes + [node] + list(reversed(right_piece.nodes))
                        paths.append(ASTPath(path, left_piece.terminal, right_piece.terminal))

    return paths


def extract_pieces(root: Node, paths: List[ASTPath], params: ExtractingParams) -> List[List[PathPiece]]:
    pieces = []
    for child in root.children:
        if child.is_leaf():
            pieces.append([PathPiece([], child)])
        else:
            pieces.append(
                [PathPiece(piece.nodes + [child], piece.terminal)
                 for child_pieces in extract_pieces(child, paths, params)
                 for piece in child_pieces if len(piece) + 1 <= params.max_length])

    paths.extend(collect_paths(pieces, root, params))

    return pieces
