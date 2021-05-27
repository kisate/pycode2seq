from torch._C import device
from code2seq.model import Code2Seq, Code2Class, TypedCode2Seq
from code2seq.utils.vocabulary import Vocabulary

import torch
from omegaconf import DictConfig, OmegaConf

from torch import nn
device = torch.device("cuda")

from os.path import join

weights_to_expand = [
    "encoder.node_embedding.weight",
]


net = torch.load("big_model.ckpt")

config = OmegaConf.load("configs/code2seq.yaml")
vocabulary_path = join(config.data_folder, config.dataset.name, config.vocabulary_name)
vocabulary = Vocabulary.load_vocabulary(vocabulary_path)
model = Code2Seq(config, vocabulary)

with open("expand.log", "w") as f:
    for key in weights_to_expand:
        weight1 = net["state_dict"][key]
        weight2 = model.state_dict()[key]
        delta = weight2.shape[0] - weight1.shape[0]
        net["state_dict"][key] = torch.cat((weight1, torch.randn(delta, weight1.shape[1]).to(device)))
        f.write(f"{key} {delta}\n")


model.load_state_dict(net["state_dict"])

torch.save({"size" : delta, "state_dict": model.state_dict()}, "new_net.ckpt")