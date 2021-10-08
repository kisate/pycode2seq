from random import shuffle

from code2seq.utils.vocabulary import Vocabulary
from mine_projects import process_holdout
from astminer_to_code2seq import preprocess_csv
from combine_vocabularies import combine_dicts
from pathlib import Path

from random import seed
import multiprocessing as mp
from tqdm import tqdm

from code2seq.preprocessing.build_vocabulary import preprocess
from omegaconf import OmegaConf

import torch

from code2seq import utils
import sys
sys.modules["utils"] = utils

if __name__ == "__main__":
    data_path = "/home/dumtrii/Documents/practos/spring2/code2vec_work/data/part"
    mined_path = Path.home() / ".cache" / "pycode2seq" / "training" / "paths"
    dataset_name = "kotlin-small"
    cli_path = "/home/dumtrii/IdeaProjects/astminer/cli.sh"
    checkpoint_path = "/home/dumtrii/Documents/practos/spring2/code2vec_work/code2seq_torch/code2seq/big_model.ckpt"

    seed(7)

    for holdout in ["val"]:
        # Path(mined_paths).mkdir(exist_ok=True, parents=True)
        # process_holdout(Path(data_path, holdout), Path(mined_path, holdout), cli_path)

        output_c2s_path = Path("data", dataset_name, f"{dataset_name}.{holdout}.c2s")

        output_c2s_path.unlink(missing_ok=True)
        output_c2s_path.parent.mkdir(exist_ok=True)

        paths = list(Path(mined_path, holdout).glob("*/"))

        # with open(output_c2s_path, "a+") as c2s_output:
        #     for project_path in tqdm(paths):
        #         try:
        #             preprocess_csv(project_path / "kt", holdout, False, c2s_output)

        #         except ValueError:
        #             print(project_path)

    config = OmegaConf.load("configs/code2seq.yaml")
    # preprocess(config)


    old_model = torch.load(checkpoint_path)
    old_voc: Vocabulary = old_model["hyper_parameters"]["vocabulary"]
    second_voc = Vocabulary.load_vocabulary(Path("data", dataset_name, "vocabulary.pkl"))

    new_voc = Vocabulary(
        old_voc.token_to_id,
        combine_dicts(old_voc.node_to_id, second_voc.node_to_id),
        second_voc.label_to_id,
    )

    new_voc.dump_vocabulary(Path("data", dataset_name, "vocabulary_new.pkl"))




