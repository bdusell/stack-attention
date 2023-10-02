from .common import BinaryStringSampler, UnmarkedBinaryVocab

class UnmarkedReverseAndCopySampler(BinaryStringSampler):

    def is_marked(self):
        return False

    def num_sections(self):
        return 3

    def binary_string_to_sample(self, w):
        return [*w, *reversed(w), *w]

UnmarkedReverseAndCopyVocab = UnmarkedBinaryVocab
