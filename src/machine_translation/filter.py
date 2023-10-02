import argparse
import pathlib
import random

import more_itertools

def exceeds_length_ratio(source, target, ratio):
    source_len = len(source)
    target_len = len(target)
    return source_len / target_len > ratio or target_len / source_len > ratio

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs=2, type=pathlib.Path, required=True)
    parser.add_argument('--outputs', nargs=2, type=pathlib.Path, required=True)
    parser.add_argument('--sample-size', type=int)
    parser.add_argument('--random-seed', type=int, default=123)
    parser.add_argument('--max-length', type=int)
    parser.add_argument('--max-length-ratio', type=float)
    args = parser.parse_args()

    for i in range(2):
        print(f'filtering {args.inputs[i]} => {args.outputs[i]}')

    def generate_filtered_pairs():
        num_examples = 0
        num_empty = 0
        num_period = 0
        num_bad_length_ratio = 0
        num_too_long = 0
        with args.inputs[0].open() as source_fin, \
             args.inputs[1].open() as target_fin:
            for pair in more_itertools.zip_equal(source_fin, target_fin):
                pair = tuple(x.rstrip('\n') for x in pair)
                stripped_pair = tuple(x.strip() for x in pair)
                if any(not x for x in stripped_pair):
                    num_empty += 1
                elif any(x == '.' for x in stripped_pair):
                    num_period += 1
                elif args.max_length_ratio is not None and exceeds_length_ratio(*pair, args.max_length_ratio):
                    num_bad_length_ratio += 1
                elif args.max_length is not None and not all(len(x) <= args.max_length for x in pair):
                    num_too_long += 1
                else:
                    yield pair
                num_examples += 1
        num_discarded = num_empty + num_period + num_bad_length_ratio + num_too_long
        print(f'total examples:     {num_examples}')
        print(f'total kept:         {num_examples - num_discarded}')
        print(f'total discarded:    {num_discarded}')
        print(f'  empty string:     {num_empty}')
        print(f'  just a period:    {num_period}')
        print(f'  bad length ratio: {num_bad_length_ratio}')
        print(f'  too long:         {num_too_long}')

    pairs = generate_filtered_pairs()
    if args.sample_size is not None:
        generator = random.Random(args.random_seed)
        pairs = generator.sample(list(pairs), args.sample_size)
        print(f'sample size:        {args.sample_size}')
    with args.outputs[0].open('w') as source_fout, \
         args.outputs[1].open('w') as target_fout:
        for source, target in pairs:
            print(source, file=source_fout)
            print(target, file=target_fout)

if __name__ == '__main__':
    main()
