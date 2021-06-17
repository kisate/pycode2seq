from argparse import ArgumentParser
from typing import Dict
from code2seq.utils.vocabulary import Vocabulary


def combine_dicts(dict1: Dict[str, int], dict2: Dict[str, int]):
    new_items = set(key for key in dict2.keys()).difference(key for key in dict1.keys())
    new_id = max(item for _, item in dict1.items())

    for item in new_items:
        dict1[item] = new_id
        new_id += 1

    return dict1


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("old_voc", type=str)
    arg_parser.add_argument("second_voc", type=str)
    arg_parser.add_argument("output", type=str)
    args = arg_parser.parse_args()

    old_voc = Vocabulary.load_vocabulary(args.old_voc)
    second_voc = Vocabulary.load_vocabulary(args.second_voc)

    new_voc = Vocabulary(
        old_voc.token_to_id,
        combine_dicts(old_voc.node_to_id, second_voc.node_to_id),
        second_voc.label_to_id,
    )

    new_voc.dump_vocabulary(args.output)
