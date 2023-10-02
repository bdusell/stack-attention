import argparse

from cfl_language_modeling.model_util import CFLModelInterface

def main():

    model_interface = CFLModelInterface(use_init=False, use_load=True, use_output=False)

    parser = argparse.ArgumentParser()
    model_interface.add_arguments(parser)
    args = parser.parse_args()

    saver = model_interface.construct_saver(args)
    model = saver.model

    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)

if __name__ == '__main__':
    main()
