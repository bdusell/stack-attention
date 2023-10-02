from ..sampler import UniformSampler

class Count3Sampler(UniformSampler):

    def num_strings_with_length(self, length):
        return int(length % 3 == 0)

    def sample(self, length, generator):
        n, r = divmod(length, 3)
        if r == 0:
            return [0]*n + [1]*n + [2]*n
        else:
            raise ValueError

class Count3Vocab:

    SYMBOLS = 'abc'

    def value(self, i):
        if 0 <= i < self.size():
            return self.SYMBOLS[i]
        else:
            raise ValueError

    def size(self):
        return len(self.SYMBOLS)
