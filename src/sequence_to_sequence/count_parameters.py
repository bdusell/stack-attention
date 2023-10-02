import argparse

import humanize

from sequence_to_sequence.data_util import add_vocabulary_arguments, load_vocabularies
from sequence_to_sequence.model_util import SequenceToSequenceModelInterface

def main():

    model_interface = SequenceToSequenceModelInterface(use_load=True, use_output=False)

    parser = argparse.ArgumentParser()
    add_vocabulary_arguments(parser)
    model_interface.add_arguments(parser)
    args = parser.parse_args()

    data = load_vocabularies(args, parser)

    saver = model_interface.construct_saver(
        args,
        source_vocab_size=len(data.source_vocab),
        target_input_vocab_size=len(data.target_input_vocab),
        target_output_vocab_size=len(data.target_output_vocab),
        tie_embeddings=data.vocab_is_shared
    )
    model = saver.model

    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)

if __name__ == '__main__':
    main()
