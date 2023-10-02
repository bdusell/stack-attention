import argparse

from cfl_language_modeling.model_util import CFLModelInterface
from cfl_language_modeling.task_util import add_task_arguments, parse_task

def main():

    model_interface = CFLModelInterface(use_load=True, use_output=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', default=False)
    model_interface.add_arguments(parser)
    add_task_arguments(parser)
    args = parser.parse_args()

    task = parse_task(parser, args)
    saver = model_interface.construct_saver(
        args,
        input_size=task.input_vocab.size(),
        output_size=task.output_vocab.size()
    )
    model = saver.model

    if args.verbose:
        for name, param in model.named_parameters():
            print(name, param.numel())
    else:
        num_params = sum(p.numel() for p in model.parameters())
        print(num_params)

if __name__ == '__main__':
    main()
