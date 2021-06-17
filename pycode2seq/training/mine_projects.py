import os
import subprocess
import functools

from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

import multiprocessing as mp

holdouts = ["train", "test", "val"]


def call_astminer(path, output_path, cli_path):
    subprocess.call(
        f"./{cli_path} code2vec --lang kt --project {path} --output {Path(output_path, path.name)} --split-tokens --granularity method --hide-method-name",
        shell=True)


def process_holdout(data_path, output_path, cli_path):
    if not Path(output_path).exists():
        os.mkdir(output_path)

    paths = list(Path(data_path).glob("*/"))

    _foo = functools.partial(call_astminer, output_path=output_path, cli_path=cli_path)
    with mp.Pool(4) as p:
        list(tqdm(p.imap(_foo, paths), total=len(paths)))


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("output", type=str)
    arg_parser.add_argument("cli_path", type=str, help="path to astminer cli.sh")

    args = arg_parser.parse_args()

    for holdout in holdouts:
        process_holdout(os.path.join(args.data, holdout), os.path.join(args.output, holdout), args.cli_path)
