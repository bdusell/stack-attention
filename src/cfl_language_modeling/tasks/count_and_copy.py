from .common import BinaryStringSampler, MarkedBinaryVocab

class CountAndCopySampler(BinaryStringSampler):

    def is_marked(self):
        return False

    def num_sections(self):
        return 3

    def binary_string_to_sample(self, w):
        x = [self.MARKER] * len(w)
        return [*w, *x, *w]

CountAndCopyVocab = MarkedBinaryVocab
