from .common import BinaryStringSampler, MarkedBinaryVocab

class MarkedCopySampler(BinaryStringSampler):

    def is_marked(self):
        return True

    def num_sections(self):
        return 2

    def binary_string_to_sample(self, w):
        return [*w, self.MARKER, *w]

MarkedCopyVocab = MarkedBinaryVocab
