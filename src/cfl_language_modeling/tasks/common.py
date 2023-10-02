import math

from ..sampler import UniformSampler

class UnmarkedVocab:

    def __init__(self, size):
        super().__init__()
        self._size = size

    def value(self, i):
        if 0 <= i < self._size:
            return str(i)
        else:
            raise ValueError

    def size(self):
        return self._size

class MarkedVocab:

    def __init__(self, size):
        super().__init__()
        self._size = size

    def value(self, i):
        if 0 <= i < self._size:
            return str(i)
        elif i == self._size:
            return '#'
        else:
            raise ValueError

    def size(self):
        return self._size + 1

class NAryStringSampler(UniformSampler):

    def __init__(self, symbol_types):
        super().__init__()
        self.symbol_types = symbol_types

    def is_marked(self):
        raise NotImplementedError

    def num_sections(self):
        raise NotImplementedError

    def nary_string_to_sample(self, w):
        raise NotImplementedError

    @property
    def marker(self):
        return self.symbol_types

    def num_markers(self):
        if self.is_marked():
            return self.num_sections() - 1
        else:
            return 0

    def divmod_length(self, length):
        return divmod(length - self.num_markers(), self.num_sections())

    def log_num_strings_with_length(self, length):
        n, r = self.divmod_length(length)
        if r == 0:
            return math.log(self.symbol_types) * n
        else:
            return -math.inf

    def sample(self, length, generator):
        n, r = self.divmod_length(length)
        if r == 0:
            w = [generator.randrange(self.symbol_types) for i in range(n)]
            return self.nary_string_to_sample(w)
        else:
            raise ValueError

class UnmarkedBinaryVocab(UnmarkedVocab):

    def __init__(self):
        super().__init__(2)

class MarkedBinaryVocab(MarkedVocab):

    def __init__(self):
        super().__init__(2)

class BinaryStringSampler(NAryStringSampler):

    def __init__(self):
        super().__init__(2)

    MARKER = 2

    def nary_string_to_sample(self, w):
        return self.binary_string_to_sample(w)

    def binary_string_to_sample(self, w):
        raise NotImplementedError
