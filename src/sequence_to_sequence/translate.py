import argparse
import pathlib

import more_itertools

from sequence_to_sequence.data_util import (
    load_prepared_data_file,
    add_vocabulary_arguments,
    load_vocabularies
)
from sequence_to_sequence.model_util import SequenceToSequenceModelInterface
from sequence_to_sequence.batcher import (
    add_batching_arguments,
    get_batcher
)

def main():

    model_interface = SequenceToSequenceModelInterface(
        use_load=True,
        use_init=False,
        use_output=False,
        require_output=False
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=pathlib.Path, required=True)
    parser.add_argument('--beam-size', type=int, required=True)
    parser.add_argument('--max-target-length', type=int, required=True)
    add_batching_arguments(parser)
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_vocabulary_arguments(parser)
    args = parser.parse_args()

    device = model_interface.get_device(args)
    sources = load_prepared_data_file(args.input)
    vocabs = load_vocabularies(args, parser)
    saver = model_interface.construct_saver(args)
    model_interface.on_before_decode(saver, [sources], args.max_target_length)
    batcher = get_batcher(parser, args, model_interface)
    batches = list(batcher.generate_source_batches(sources))
    del batcher
    ordered_outputs = [None] * len(sources)
    for batch in batches:
        source = model_interface.prepare_source([s for i, s in batch], device, vocabs)
        output = model_interface.decode(
            model=saver.model,
            model_source=source,
            bos_symbol=vocabs.target_input_vocab.bos_index,
            beam_size=args.beam_size,
            eos_symbol=vocabs.target_output_vocab.eos_index,
            max_length=args.max_target_length
        )
        for (i, s), output_sequence in more_itertools.zip_equal(batch, output):
            ordered_outputs[i] = output_sequence
    for output_sequence in ordered_outputs:
        print(' '.join(vocabs.target_output_vocab.to_string(w) for w in output_sequence))

if __name__ == '__main__':
    main()
