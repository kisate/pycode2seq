from argparse import ArgumentParser
from os import path
from random import seed

from code2seq.preprocessing.astminer_to_code2seq import preprocess_csv

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("--shuffle", action="store_true")
    args = arg_parser.parse_args()
    data_path = path.join("data", args.data)

    seed(7)
    for holdout in ["train", "val", "test"]:
        print(f"preprocessing {holdout} data")
        preprocess_csv("data", args.data, holdout, args.shuffle)
