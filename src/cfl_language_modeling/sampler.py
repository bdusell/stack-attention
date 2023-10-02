import functools
import math

from lib.lang_algorithm.parsing import Parser
from lib.lang_algorithm.pcfg import string_log_probability

class Sampler:

    def valid_lengths(self, min_length, max_length):
        raise NotImplementedError

    def sample(self, length, generator):
        raise NotImplementedError

    def log_probability_given_length(self, length):
        raise NotImplementedError

class PCFGSampler(Sampler):

    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def valid_lengths(self, min_length, max_length):
        return self.sampler.valid_lengths((min_length, max_length))

    def sample(self, length, generator):
        return self.sampler.sample(length, generator)

    def log_probability_given_length(self, sample):
        length = len(sample)
        gw_log_prob = string_log_probability(self.parser, sample)
        summed_gw_prob = self.sampler.get_inside_probability(self.grammar.start, length)
        summed_gw_log_prob = math.log(summed_gw_prob)
        return gw_log_prob - summed_gw_log_prob

    @property
    def grammar(self):
        return self.sampler.grammar

    @functools.cached_property
    def parser(self):
        return Parser(self.grammar)

class UniformSampler(Sampler):

    def valid_lengths(self, min_length, max_length):
        return [l for l in range(min_length, max_length+1) if self.is_valid_length(l)]

    def is_valid_length(self, length):
        return self.log_num_strings_with_length(length) > -math.inf

    def log_num_strings_with_length(self, length):
        raise NotImplementedError

    def sample(self, length, generator):
        raise NotImplementedError

    def log_probability_given_length(self, sample):
        return -self.log_num_strings_with_length(len(sample))
