import os
from random import shuffle
from code2seq.data.path_context_dataset import PathContextDataset

from pycode2seq.inference.parsing.utils import read_astminer
from typing import List, Tuple, Dict

import torch

from code2seq.data.path_context import BatchedLabeledPathContext, LabeledPathContext, Path
from code2seq.model import Code2Seq
from omegaconf import OmegaConf
from torch import Tensor

from pycode2seq.inference.language import Language
from pycode2seq.inference.model.labels import LabeledData, extract_labels_with_paths
from pycode2seq.inference.model.loader import ModelLoader
from pycode2seq.inference.paths.extracting import ExtractingParams
from pycode2seq.inference.model.data import OldVocabulary


class Model:
    models_gdrive_file_ids = {
        "kt_java": "1v8GFPraNFLmiQxXBZAK3K9CIyhIADp-t",
        "java": "1v8GFPraNFLmiQxXBZAK3K9CIyhIADp-t",
        "kt": "1v8GFPraNFLmiQxXBZAK3K9CIyhIADp-t",
        "kotlin": "1v8GFPraNFLmiQxXBZAK3K9CIyhIADp-t"
    }

    multi_models = ["kt_java"]

    @staticmethod
    def load(name: str) -> 'Model':
        save_path = ModelLoader.model_path(name)
        os.makedirs(save_path, exist_ok=True)

        model_path = os.path.join(save_path, "model")
        config_path = os.path.join(model_path, "code2seq.yaml")
        vocabulary_path = os.path.join(model_path, "vocabulary.pkl")
        checkpoint_path = os.path.join(model_path, "model.ckpt")

        required_files = [config_path, vocabulary_path, checkpoint_path]

        if not all(os.path.exists(file) for file in required_files):
            print("Downloading model")
            ModelLoader.load(name, Model.models_gdrive_file_ids[name])

        return Model(config_path, vocabulary_path, checkpoint_path, ExtractingParams(8, 3, 200), name)

    def __init__(self, config_path: str, vocabulary_path: str, checkpoint_path: str,
                 extracting_params: ExtractingParams, model_name: str) -> None:
        self.config = OmegaConf.load(config_path)
        self.vocabulary = OldVocabulary(vocabulary_path)
        self.model = Code2Seq(self.config.model, self.config.optimizer, self.vocabulary)

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

        self.device = torch.device("cpu")
        self.to(self.device)
        self.model.eval()

        self.extracting_params = extracting_params

        self.model_name = model_name
        self.default_lang = None if model_name in Model.multi_models else model_name

    def to(self, device: torch.device):
        self.device = device
        self.model.to(self.device)

    def _prepare_batches(self, file_path: str, language: Language) -> Tuple[List[BatchedLabeledPathContext], List[str]]:
        root = language.parse(file_path)
        data = extract_labels_with_paths(root, self.extracting_params, language.split_on_methods)
        method_names = [d.method_name for d in data]
        batches = [BatchedLabeledPathContext([self._labeled_data_to_sample(method, 200, True)]) for method in data]

        for batch in batches:
            batch.move_to_device(self.device)

        return batches, method_names

    def methods_embeddings(self, file_path: str, language: str = None) -> Dict[str, Tensor]:
        language = language or self.default_lang
        if language is None:
            language = file_path.split('.')[-1]
        batches, method_names = self._prepare_batches(file_path, Language.by_name(language))

        with torch.no_grad():
            embeddings = {}
            for batch, method_name in zip(batches, method_names):
                encoded_paths = self.model._encoder(batch.from_token, batch.path_nodes, batch.to_token)
                # [n layers; batch size; decoder size]
                coded_batch = [ctx_batch.mean(0).unsqueeze(0) for ctx_batch in
                               encoded_paths.split(batch.contexts_per_label)]
                initial_state = (torch.cat(coded_batch).unsqueeze(0))
                embeddings[method_name] = initial_state.squeeze()
            return embeddings

    def run_model_on_astminer_csv(self, data_path: str, language: str) -> List[Tensor]:
        # data_path -- path to folder with generated csvs
        data = read_astminer(data_path)
        batches = [PathContextBatch([self._string_to_sample(method, 200, True)]) for method in data]

        for batch in batches:
            batch.move_to_device(self.device)

        with torch.no_grad():
            return [self.model(batch.contexts, batch.contexts_per_label, batch.labels.shape[0]) for batch in batches]

    def run_model_on_file(self, file_path: str, language: str) -> List[Tensor]:
        batches, method_names = self._prepare_batches(file_path, Language.by_name(language))

        with torch.no_grad():
            return [self.model(batch.contexts, batch.contexts_per_label, batch.labels.shape[0]) for batch in batches]

    def _run_model_on_file_with_metrics(self, file_path: str, language: str):
        batches, method_names = self._prepare_batches(file_path, Language.by_name(language))

        results = []

        with torch.no_grad():
            for batch in batches:
                logits = self.model(batch.contexts, batch.contexts_per_label, batch.labels.shape[0], batch.labels)
                prediction = logits.argmax(-1)

                statistic = PredictionStatistic(True, self.model._label_pad_id, self.model._metric_skip_tokens)
                batch_metric = statistic.update_statistic(batch.labels, prediction)

                results.append(batch_metric)

        return results

    def _string_to_sample(self, data: str, max_contexts: int, random_context: bool) -> LabeledPathContext:
        str_label, *str_path_contexts = data.split()
        if str_label == "" or len(str_path_contexts) == 0:
            print(f"Bad sample {data}")
            return None

        # choose random paths
        n_contexts = min(len(str_path_contexts), max_contexts)
        if random_context:
            shuffle(str_path_contexts)
        str_contexts = str_path_contexts[:n_contexts]
        
        parameters = self.config.data

        # convert string label to wrapped numpy array
        label = PathContextDataset.tokenize_label(str_label, self.vocabulary.label_to_id, parameters.max_label_parts)

        # convert each context to list of ints and then wrap into numpy array
        paths = [self._string_to_path(path.split(","), parameters) for path in str_path_contexts]

        return LabeledPathContext(label, paths)
    
    def _string_to_path(self, data: List[str], parameters) -> Path:
        return Path(
            from_token=PathContextDataset.tokenize_token(data[0], self.vocabulary.token_to_id, parameters.max_token_parts),
            path_node=PathContextDataset.tokenize_token(data[1], self.vocabulary.node_to_id, parameters.path_length),
            to_token=PathContextDataset.tokenize_token(data[2], self.vocabulary.token_to_id, parameters.max_token_parts),
        )

    def _labeled_data_to_sample(self, data: LabeledData, max_contexts: int, random_context: bool) -> LabeledPathContext:
        return self._string_to_sample(str(data), max_contexts, random_context)

    def _get_label_by_id(self, id: int) -> str:
        return list(self.vocabulary.label_to_id.keys())[list(self.vocabulary.label_to_id.values()).index(id)]

    def prediction_to_text(self, prediction: Tensor) -> str:
        ids = prediction.argmax(-1)
        return "|".join([self._get_label_by_id(id[0]) for id in ids])
