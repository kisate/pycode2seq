from argparse import ArgumentParser
from os.path import join

import torch

from omegaconf import OmegaConf
from code2seq.model import Code2Seq
from code2seq.utils.vocabulary import Vocabulary


weights_to_expand = [
    "encoder.node_embedding.weight",
]


def expand_weights(checkpoint_path, config_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = torch.load(checkpoint_path)

    config = OmegaConf.load(config_path)
    vocabulary_path = join(config.data_folder, config.dataset.name, config.vocabulary_name)
    vocabulary = Vocabulary.load_vocabulary(vocabulary_path)
    model = Code2Seq(config, vocabulary)

    deltas = {}

    for key in weights_to_expand:
        weight1 = net["state_dict"][key]
        weight2 = model.state_dict()[key]

        delta = weight2.shape[0] - weight1.shape[0]
        net["state_dict"][key] = torch.cat((weight1, torch.randn(delta, weight1.shape[1]).to(device)))

        deltas[key] = delta

    model.load_state_dict(net["state_dict"])

    torch.save({"deltas": deltas, "state_dict": model.state_dict()}, output_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str)
    arg_parser.add_argument("config", type=str)
    arg_parser.add_argument("output", type=str)
    args = arg_parser.parse_args()

    expand_weights(args.checkpoint, args.config, args.output)
