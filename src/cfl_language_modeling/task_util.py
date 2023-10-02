import attr

from utils.cli_util import get_kwargs
from lib.formal_models.pcfg_length_sampling import LengthSampler
from lib.formal_models.pcfg_tools import (
    remove_epsilon_rules, remove_unary_rules)
from .grammars.marked_reversal import (
    MarkedReversalGrammar, MarkedReversalVocab)
from .grammars.unmarked_reversal import (
    UnmarkedReversalGrammar, UnmarkedReversalVocab)
from .grammars.padded_reversal import (
    PaddedReversalGrammar, PaddedReversalVocab)
from .grammars.dyck import (
    DyckGrammar, DyckVocab)
from .grammars.hardest_cfl import (
    HardestCFLGrammar, HardestCFLVocab)
from .tasks.count_3 import (
    Count3Sampler, Count3Vocab)
from .tasks.marked_copy import (
    MarkedCopySampler, MarkedCopyVocab)
from .tasks.unmarked_copy import (
    UnmarkedCopySampler, UnmarkedCopyVocab)
from .tasks.marked_reverse_and_copy import (
    MarkedReverseAndCopySampler, MarkedReverseAndCopyVocab)
from .tasks.unmarked_reverse_and_copy import (
    UnmarkedReverseAndCopySampler, UnmarkedReverseAndCopyVocab)
from .tasks.count_and_copy import (
    CountAndCopySampler, CountAndCopyVocab)
from .tasks.unmarked_copy_different_alphabets import (
    UnmarkedCopyDifferentAlphabetsSampler, UnmarkedCopyDifferentAlphabetsVocab)
from .sampler import PCFGSampler

def add_task_arguments(parser):
    group = parser.add_argument_group('Task options')
    group.add_argument('--task',
        choices=[
            'marked-reversal',
            'unmarked-reversal',
            'padded-reversal',
            'dyck',
            'hardest-cfl',
            'count-3',
            'marked-copy',
            'unmarked-copy',
            'marked-reverse-and-copy',
            'unmarked-reverse-and-copy',
            'count-and-copy',
            'unmarked-copy-different-alphabets'
        ],
        required=True,
        help='Choice of task. Possible choices are marked-reversal, '
             'unmarked-reversal, padded-reversal, dyck, and hardest-cfl.')
    # marked-reversal, unmarked-reversal
    group.add_argument('--symbol-types', type=int,
        default=2,
        help='(marked-reversal, unmarked-reversal) The number of symbol types '
             'in the reversed string.')
    group.add_argument('--mean-length', type=float,
        default=60,
        help='(marked-reversal, unmarked-reversal) Mean length of the '
             'reversed string.')
    # padded-reversal
    group.add_argument('--mean-content-length', type=float,
        default=60,
        help='(padded-reversal) Mean length of the reversed string.')
    group.add_argument('--mean-padding-length', type=float,
        default=30,
        help='(padded-reversal) Mean length of the padding in the middle.')
    # dyck
    group.add_argument('--bracket-types', type=int,
        default=2,
        help='(dyck) Number of bracket pair types.')
    group.add_argument('--mean-bracket-splits', type=float,
        default=1,
        help='(dyck) Mean number of bracket splits.')
    group.add_argument('--mean-nesting-depth', type=float,
        default=40,
        help='(dyck) Mean bracket nesting depth.')
    # hardest-cfl
    group.add_argument('--mean-num-commas', type=float,
        default=0.5,
        help='(hardest-cfl) Mean number of commas in each decoy string.')
    group.add_argument('--mean-short-filler-length', type=float,
        default=0.5,
        help='(hardest-cfl) Mean length of short filler strings in decoys.')
    group.add_argument('--mean-long-filler-length', type=float,
        default=2,
        help='(hardest-cfl) Mean length of long filler strings in decoys.')
    group.add_argument('--semicolon-probability', type=float,
        default=0.25,
        help='(hardest-cfl) Probability of generating a split point with a '
             'semicolon in between every symbol of the Dyck string.')

@attr.s
class Task:
    sampler = attr.ib()
    input_vocab = attr.ib()
    output_vocab = attr.ib()

def parse_task(parser, args):
    name = args.task
    if name == 'count-3':
        sampler = Count3Sampler()
        vocab = Count3Vocab()
    elif name == 'marked-copy':
        sampler = MarkedCopySampler()
        vocab = MarkedCopyVocab()
    elif name == 'unmarked-copy':
        sampler = UnmarkedCopySampler()
        vocab = UnmarkedCopyVocab()
    elif name == 'marked-reverse-and-copy':
        sampler = MarkedReverseAndCopySampler(args.symbol_types)
        vocab = MarkedReverseAndCopyVocab(args.symbol_types)
    elif name == 'unmarked-reverse-and-copy':
        sampler = UnmarkedReverseAndCopySampler()
        vocab = UnmarkedReverseAndCopyVocab()
    elif name == 'count-and-copy':
        sampler = CountAndCopySampler()
        vocab = CountAndCopyVocab()
    elif name == 'unmarked-copy-different-alphabets':
        sampler = UnmarkedCopyDifferentAlphabetsSampler()
        vocab = UnmarkedCopyDifferentAlphabetsVocab()
    else:
        if name == 'marked-reversal':
            grammar = MarkedReversalGrammar(**get_kwargs(parser, args, [
                'symbol_types',
                'mean_length'
            ]))
            vocab = MarkedReversalVocab(args.symbol_types)
        elif name == 'unmarked-reversal':
            grammar = UnmarkedReversalGrammar(**get_kwargs(parser, args, [
                'symbol_types',
                'mean_length'
            ]))
            vocab = UnmarkedReversalVocab(args.symbol_types)
        elif name == 'padded-reversal':
            grammar = PaddedReversalGrammar(**get_kwargs(parser, args, [
                'symbol_types',
                'mean_content_length',
                'mean_padding_length'
            ]))
            vocab = PaddedReversalVocab(args.symbol_types)
        elif name == 'dyck':
            grammar = DyckGrammar(**get_kwargs(parser, args, [
                'bracket_types',
                'mean_bracket_splits',
                'mean_nesting_depth'
            ]))
            vocab = DyckVocab(args.bracket_types)
        elif name == 'hardest-cfl':
            grammar = HardestCFLGrammar(**get_kwargs(parser, args, [
                'mean_num_commas',
                'mean_short_filler_length',
                'mean_long_filler_length',
                'semicolon_probability',
                'mean_bracket_splits',
                'mean_nesting_depth'
            ]))
            vocab = HardestCFLVocab()
        else:
            raise ValueError
        sampler = construct_pcfg_sampler(grammar)
    output_vocab = EndSymbolVocab(vocab)
    return Task(sampler, vocab, output_vocab)

def construct_pcfg_sampler(grammar):
    # The length sampler requires epsilon and unary rules to be removed.
    remove_epsilon_rules(grammar)
    remove_unary_rules(grammar)
    length_sampler = LengthSampler(grammar)
    return PCFGSampler(length_sampler)

class EndSymbolVocab:

    def __init__(self, vocab, end_symbol_string='</s>'):
        super().__init__()
        self.vocab = vocab
        self.end_symbol = self.vocab.size()
        self.end_symbol_string = end_symbol_string

    def value(self, i):
        if i == self.end_symbol:
            return self.end_symbol_string
        else:
            return self.vocab.value(i)

    def size(self):
        return self.end_symbol + 1
