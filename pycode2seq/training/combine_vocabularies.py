from typing import Dict
from code2seq.utils.vocabulary import Vocabulary

from os import path

data_path = "data/kotlin-small"

java_voc = Vocabulary.load_vocabulary("big_voc.pkl")
kotlin_voc = Vocabulary.load_vocabulary(path.join(data_path, "vocabulary.pkl"))

def combine_dicts(dict1: Dict[str, int], dict2: Dict[str, int]):
    new_items = set(key for key in dict2.keys()).difference(key for key in dict1.keys())
    print(f"{len(new_items)} new items")
    print(f"{len(dict1)} old items")
    new_id = max(item for _, item in dict1.items())
    for item in new_items:
        dict1[item] = new_id
        new_id += 1
    return dict1

new_voc = Vocabulary(
    java_voc.token_to_id,
    combine_dicts(java_voc.node_to_id, kotlin_voc.node_to_id),
    java_voc.label_to_id
    )

new_voc.dump_vocabulary(path.join(data_path, "vocabulary_new.pkl"))
