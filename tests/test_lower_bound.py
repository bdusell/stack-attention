import math
import random
import unittest

from lib.formal_models.parse_tree import ParseTree as Tree
from lib.formal_models.pcfg import Grammar, Rule, Terminal, Nonterminal
from lib.formal_models.pcfg_length_sampling import LengthSampler
from lib.lang_algorithm.parsing import Parser
from lib.lang_algorithm.pcfg import string_log_probability

S = Nonterminal('S')
a = Terminal('a')

class TestLowerBoundOnGrammar1(unittest.TestCase):

    def construct_grammar(self):
        return Grammar(S, [
            Rule(S, [a], 1.0)
        ])

    def test_inside_probs(self):
        G = self.construct_grammar()
        sampler = LengthSampler(G)
        self.assertAlmostEqual(sampler.get_inside_probability(S, 0), 0.0)
        self.assertAlmostEqual(sampler.get_inside_probability(S, 1), 1.0)
        self.assertAlmostEqual(sampler.get_inside_probability(S, 2), 0.0)

    def test_lang_parser(self):
        G = self.construct_grammar()
        sampler = LengthSampler(G)
        parser = Parser(G)
        sample = ['a']
        expected_parses = { Tree(S, [Tree(a)]) }
        actual_parses = set(parser.parse(sample))
        self.assertEqual(actual_parses, expected_parses)

    def test_parse_probability(self):
        G = self.construct_grammar()
        sampler = LengthSampler(G)
        parser = Parser(G)
        sample = ['a']
        log_prob = string_log_probability(parser, sample)
        prob = math.exp(log_prob)
        self.assertAlmostEqual(prob, 1.0)

class TestLowerBoundOnGrammar2(unittest.TestCase):

    def construct_grammar(self):
        return Grammar(S, [
            Rule(S, [S, S], 0.5),
            Rule(S, [a], 0.5)
        ])

    def test_inside_probs(self):
        G = self.construct_grammar()
        sampler = LengthSampler(G)
        self.assertAlmostEqual(sampler.get_inside_probability(S, 0), 0.0)
        self.assertAlmostEqual(sampler.get_inside_probability(S, 1), 0.5)
        self.assertAlmostEqual(sampler.get_inside_probability(S, 2), 0.5 ** 3)
        self.assertAlmostEqual(
            sampler.get_inside_probability(S, 3),
            0.5 * 2 * 0.5**4
        )

    def test_lang_parser(self):
        G = self.construct_grammar()
        sampler = LengthSampler(G)
        parser = Parser(G)
        sample = ['a']
        expected_parses = { Tree(S, [Tree(a)]) }
        actual_parses = set(parser.parse(sample))
        self.assertEqual(actual_parses, expected_parses)
        sample = ['a', 'a']
        expected_parses = { Tree(S, [Tree(S, [Tree(a)]), Tree(S, [Tree(a)])]) }
        actual_parses = set(parser.parse(sample))
        self.assertEqual(actual_parses, expected_parses)
        sample = ['a', 'a', 'a']
        expected_parses = {
            Tree(S, [
                Tree(S, [Tree(a)]),
                Tree(S, [
                    Tree(S, [Tree(a)]),
                    Tree(S, [Tree(a)])
                ])
            ]),
            Tree(S, [
                Tree(S, [
                    Tree(S, [Tree(a)]),
                    Tree(S, [Tree(a)])
                ]),
                Tree(S, [Tree(a)])
            ])
        }
        actual_parses = set(parser.parse(sample))
        self.assertEqual(actual_parses, expected_parses)

    def test_parse_probability(self):
        G = self.construct_grammar()
        sampler = LengthSampler(G)
        parser = Parser(G)
        sample = ['a', 'a', 'a']
        log_prob = string_log_probability(parser, sample)
        prob = math.exp(log_prob)
        self.assertAlmostEqual(
            prob,
            0.5 * 2 * 0.5**4
        )

if __name__ == '__main__':
    unittest.main()
