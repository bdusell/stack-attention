from .common import NAryStringSampler, MarkedVocab

class MarkedReverseAndCopySampler(NAryStringSampler):

    def is_marked(self):
        return True

    def num_sections(self):
        return 3

    def nary_string_to_sample(self, w):
        return [*w, self.marker, *reversed(w), self.marker, *w]

MarkedReverseAndCopyVocab = MarkedVocab
