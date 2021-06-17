from argparse import ArgumentParser
from io import FileIO
from os import path, remove
from typing import Dict

import numpy
from tqdm import tqdm

from random import shuffle, seed

from pathlib import Path

import multiprocessing as mp


def _get_id2value_from_csv(path_: str) -> Dict[str, str]:
    return dict(numpy.genfromtxt(path_, delimiter=",", dtype=(str, str))[1:])


def preprocess_csv(data_folder: str, holdout_name: str, is_shuffled: bool, c2s_output: FileIO):
    """
    Preprocessing for files tokens.csv, paths.csv, node_types.csv
    """
    id_to_token_data_path = path.join(data_folder, f"tokens.csv")
    id_to_type_data_path = path.join(data_folder, f"node_types.csv")
    id_to_paths_data_path = path.join(data_folder, f"paths.csv")
    path_contexts_path = path.join(data_folder, f"path_contexts.csv")

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
            output_lines.append(" ".join(parsed_line + ["\n"]))
        if is_shuffled:
            shuffle(output_lines)
        c2s_output.write("".join(output_lines))


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("output", type=str)
    arg_parser.add_argument("holdout", type=str)
    arg_parser.add_argument("--shuffle", action="store_true")
    args = arg_parser.parse_args()

    seed(7)

    output_c2s_path = path.join(args.output, f"data.{args.holdout}.c2s")

    if path.exists(output_c2s_path):
        remove(output_c2s_path)

    paths = list(Path(args.data).glob("*/"))

    with open(output_c2s_path, "a+") as c2s_output, mp.Pool(4) as p:
        for project_path in tqdm(paths):
            try:
                preprocess_csv(path.join(str(project_path), "kt"), args.holdout, args.shuffle, c2s_output)

            except ValueError:
                print(project_path)
