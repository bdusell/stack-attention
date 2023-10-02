from .common import BinaryStringSampler, UnmarkedVocab

class UnmarkedCopyDifferentAlphabetsSampler(BinaryStringSampler):

    def is_marked(self):
        return False

    def num_sections(self):
        return 2

    def binary_string_to_sample(self, w):
        return [*w, *[c + 2 for c in w]]

class UnmarkedCopyDifferentAlphabetsVocab(UnmarkedVocab):

    def __init__(self):
        super().__init__(4)
