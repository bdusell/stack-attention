import torch

from vocab2 import ToIntVocabularyBuilder
from language_modeling.model_util import LanguageModelingModelInterface
from language_modeling.data_util import add_vocabulary_arguments, load_vocabularies
from sequence_to_sequence.model_util import parse_layers
from cfl_language_modeling.plot_stack_attention_heatmap import run_main

def convert_tokens_to_input_tensor(model_interface, vocabs, device, input_string_strs):
    input_vocab = vocabs[1].input_vocab
    input_string_ints = [input_vocab.to_int(s) for s in input_string_strs]
    input_tensor, _ = model_interface.prepare_batch(
        [torch.tensor(input_string_ints)],
        device,
        vocabs[0]
    )
    return input_tensor

def main():
    run_main(
        model_interface_class=LanguageModelingModelInterface,
        add_vocab_arguments=add_vocabulary_arguments,
        load_vocab=lambda parser, args: (
            # TODO Loading this multiple times is inefficient
            load_vocabularies(args, parser),
            load_vocabularies(args, parser, ToIntVocabularyBuilder())
        ),
        construct_saver=lambda model_interface, args, vocabs: model_interface.construct_saver(
            args,
            input_size=len(vocabs[0].input_vocab),
            output_size=len(vocabs[0].output_vocab)
        ),
        get_layers=lambda saver: list(parse_layers(saver.kwargs['layers'])),
        convert_tokens_to_input_tensor=convert_tokens_to_input_tensor,
        format_symbol_label=lambda s, use_safe_latex: s
    )

if __name__ == '__main__':
    main()
