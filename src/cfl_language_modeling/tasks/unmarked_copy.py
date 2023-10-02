from .common import BinaryStringSampler, UnmarkedBinaryVocab

class UnmarkedCopySampler(BinaryStringSampler):

    def is_marked(self):
        return False

    def num_sections(self):
        return 2

    def binary_string_to_sample(self, w):
        return [*w, *w]

UnmarkedCopyVocab = UnmarkedBinaryVocab
