import pickle

from code2seq.data.vocabulary import Vocabulary


class OldVocabulary(Vocabulary):
    def __init__(self, vocabulary_file: str):
        with open(vocabulary_file, "rb") as f:
            vocabulary = pickle.load(f)
            self._label_to_id = vocabulary[self.LABEL]

            self._token_to_id = vocabulary[self.TOKEN]

            self._node_to_id = vocabulary[self.NODE]