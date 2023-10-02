import argparse
import pathlib

import matplotlib.pyplot as plt

from sequence_to_sequence.data_util import load_prepared_data_file

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=pathlib.Path, required=True)
    args = parser.parse_args()

    data = load_prepared_data_file(args.input)

    fig, ax = plt.subplots()
    ax.set_xlabel('Length')
    ax.set_ylabel('Count')
    ax.hist([len(x) for x in data], bins='auto')
    plt.show()

if __name__ == '__main__':
    main()
