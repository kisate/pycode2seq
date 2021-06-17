from typing import List
from pycode2seq.inference.utils import download_file_from_google_drive, extract_archive
from pycode2seq.inference.methods.splitting_java import split_java_into_methods
from pycode2seq.inference.methods.splitting_kotlin import split_kotlin_into_methods
import torch
from torch.functional import Tensor
from pycode2seq.inference.paths.extracting import ExtractingParams
from antlr4 import *
from pycode2seq.inference.labels import LabeledData, extract_labels_with_paths 

from pycode2seq.inference.parsing.languages import parse_java_file, parse_kotlin_file

from omegaconf import OmegaConf
from code2seq.utils.vocabulary import Vocabulary
from code2seq.model import Code2Seq

from code2seq.dataset.data_classes import ContextPart, FROM_TOKEN, PATH_NODES, PathContextSample, PathContextBatch, TO_TOKEN
from code2seq.dataset.path_context_dataset import PathContextDataset
from code2seq.utils.converting import strings_to_wrapped_numpy

from code2seq.utils.metrics import PredictionStatistic

import numpy as np

import os

from code2seq import utils
import sys
sys.modules["utils"] = utils

lang_dict = {
    "kt" : (parse_kotlin_file, split_kotlin_into_methods),
    "java" : (parse_java_file, split_java_into_methods)
}

def split_file_into_labeled_data(file_path: str, params: ExtractingParams, parse_file, split_into_methods) -> List[LabeledData]:
    root = parse_file(file_path)
    return extract_labels_with_paths(root, params, split_into_methods)

class ModelRunner:
    def __init__(self, config_path: str, vocabulary_path: str, checkpoint_path: str, extracting_params: ExtractingParams) -> None:
        self.config = OmegaConf.load(config_path)
        self.vocabulary = Vocabulary.load_vocabulary(vocabulary_path)
        self.model = Code2Seq(self.config, self.vocabulary)

        self.model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.extracting_params = extracting_params

    def get_embedding(self, batch: PathContextBatch) -> Tensor:
        encoded_paths = self.model.encoder(batch.contexts)
       
        # [n layers; batch size; decoder size]
        initial_state = (
            torch.cat([ctx_batch.mean(0).unsqueeze(0) for ctx_batch in encoded_paths.split(batch.contexts_per_label)])
            .unsqueeze(0)
        )
        return initial_state

    def run_embeddings_on_file(self, file_path: str, language: str) -> List[Tensor]:
        data = split_file_into_labeled_data(file_path, self.extracting_params, *lang_dict[language])
        batches = [PathContextBatch([self.labeled_data_to_sample(method, 200, True)]) for method in data]

        for batch in batches:
            batch.move_to_device(self.device)

        with torch.no_grad():
            return [self.get_embedding(batch) for batch in batches]

    def run_model_on_file(self, file_path: str, language: str) -> List[Tensor]:
        data = split_file_into_labeled_data(file_path, self.extracting_params, *lang_dict[language])
        batches = [PathContextBatch([self.labeled_data_to_sample(method, 200, True)]) for method in data]

        for batch in batches:
            batch.move_to_device(self.device)

        with torch.no_grad():
            return [self.model(batch.contexts, batch.contexts_per_label, batch.labels.shape[0]) for batch in batches]

    def run_model_on_file_with_metrics(self, file_path: str, language: str):
        data = split_file_into_labeled_data(file_path, self.extracting_params, *lang_dict[language])
        batches = [PathContextBatch([self.labeled_data_to_sample(method, 200, True)]) for method in data]
        for batch in batches:
            batch.move_to_device(self.device)

        results = []

        with torch.no_grad():
            for batch in batches:
                logits = self.model(batch.contexts, batch.contexts_per_label, batch.labels.shape[0], batch.labels)
                prediction = logits.argmax(-1)

                statistic = PredictionStatistic(True, self.model._label_pad_id, self.model._metric_skip_tokens)
                batch_metric = statistic.update_statistic(batch.labels, prediction)

                results.append(batch_metric)
        
        return results

    def labeled_data_to_sample(self, data: LabeledData, max_contexts: int, random_context: bool) -> PathContextSample:
        n_contexts = min(len(data.paths), max_contexts)
        context_indexes = np.arange(n_contexts)
        if random_context:
            np.random.shuffle(context_indexes)

        parameters = self.config.dataset.target

        # convert string label to wrapped numpy array
        wrapped_label = strings_to_wrapped_numpy(
            [data.label],
            self.vocabulary.label_to_id,
            parameters.is_splitted,
            parameters.max_parts,
            parameters.is_wrapped,
        )

        
        context_parts = [
            ContextPart(FROM_TOKEN, self.vocabulary.token_to_id, self.config.dataset.token),
            ContextPart(PATH_NODES, self.vocabulary.node_to_id, self.config.dataset.path),
            ContextPart(TO_TOKEN, self.vocabulary.token_to_id, self.config.dataset.token),
        ]

        # convert each context to list of ints and then wrap into numpy array
        splitted_contexts = [PathContextDataset._split_context(str(data.paths[i])) for i in context_indexes]
        contexts = {}
        for _cp in context_parts:
            str_values = [_sc[_cp.name] for _sc in splitted_contexts]
            contexts[_cp.name] = strings_to_wrapped_numpy(
                str_values, _cp.to_id, _cp.parameters.is_splitted, _cp.parameters.max_parts, _cp.parameters.is_wrapped
            )

        return PathContextSample(contexts=contexts, label=wrapped_label, n_contexts=n_contexts)

    def get_label_by_id(self, id: int) -> str:
        return list(self.vocabulary.label_to_id.keys())[list(self.vocabulary.label_to_id.values()).index(id)]

    def prediction_to_text(self, prediction: Tensor) -> str:
        ids = prediction.argmax(-1)
        return "|".join([ self.get_label_by_id(id[0]) for id in ids])


class DefaultModelRunner(ModelRunner):
    FILE_ID = "1v8GFPraNFLmiQxXBZAK3K9CIyhIADp-t"
    
    def __init__(self, save_path) -> None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        download_path = os.path.join(save_path, "model.tar.xz")
        model_path = os.path.join(save_path, "model")
        config_path = os.path.join(model_path, "code2seq.yaml")
        vocabulary_path = os.path.join(model_path, "vocabulary.pkl")
        checkpoint_path = os.path.join(model_path, "model.ckpt")

        if not os.path.exists(model_path):
            print("Model not found. Downloading")
            download_file_from_google_drive(self.FILE_ID, download_path)
            extract_archive(download_path, save_path)


        super().__init__(config_path, vocabulary_path, checkpoint_path, ExtractingParams(8, 3, 200))