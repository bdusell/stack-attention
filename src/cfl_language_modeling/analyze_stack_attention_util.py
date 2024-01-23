import argparse
import pathlib

import torch
import torch_semiring_einsum

from stack_rnn_models.nondeterministic_stack import logits_to_actions

class AnalyzeStackAttention:

    def get_model_interface_class(self):
        raise NotImplementedError

    def add_vocab_arguments(self, parser):
        raise NotImplementedError

    def add_arguments(self, parser):
        raise NotImplementedError

    def load_vocab(self, parser, args):
        raise NotImplementedError

    def construct_saver(self, model_interface, args, vocabs):
        raise NotImplementedError

    def get_layers(self, saver):
        raise NotImplementedError

    def convert_tokens_to_input_tensor(self, model_interface, vocabs, device, input_string_strs):
        raise NotImplementedError

    def run(self, args, stack_attention_type, input_token_strs, actions):
        raise NotImplementedError

    def main(self):

        model_interface_class = self.get_model_interface_class()
        model_interface = model_interface_class(use_load=True, use_init=False, use_output=False)

        parser = argparse.ArgumentParser()
        model_interface.add_arguments(parser)
        self.add_vocab_arguments(parser)
        parser.add_argument('--input-string', type=pathlib.Path, required=True)
        self.add_arguments(parser)
        parser.add_argument('--nd-actions', choices=['raw', 'sum', 'normalize'], default='raw')
        args = parser.parse_args()

        device = model_interface.get_device(args)
        vocabs = self.load_vocab(parser, args)

        saver = self.construct_saver(model_interface, args, vocabs)
        model = saver.model

        transformer_layers = self.get_layers(saver)
        if not transformer_layers:
            raise ValueError
        stack_attention_layers = [
            (name, layer_args)
            for name, layer_args in transformer_layers
            if name != 'transformer'
        ]
        stack_attention_types = [name for name, layer_args in stack_attention_layers]
        if stack_attention_types not in (['superposition'], ['nondeterministic']):
            raise ValueError
        (stack_attention_type, stack_attention_args), = stack_attention_layers

        input_string_strs = read_input_tokens(args.input_string)
        input_tensor = self.convert_tokens_to_input_tensor(
            model_interface,
            vocabs,
            device,
            input_string_strs
        )

        with torch.no_grad():
            model.eval()
            result = model(
                input_tensor,
                tag_kwargs=dict(
                    superposition=dict(
                        return_actions=True
                    ),
                    nondeterministic=dict(
                        block_size=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE,
                        return_actions=True
                    )
                )
            )
            raw_actions = result.extra_outputs[0][:, 0]
            if stack_attention_type == 'superposition':
                # raw_actions : sequence_length x 3
                # Reorder from push, pop, no-op to push, no-op, pop
                actions = raw_actions[:, (0, 2, 1)]
            elif stack_attention_type == 'nondeterministic':
                for name, layer_args in transformer_layers:
                    if name == 'nondeterministic':
                        break
                num_states, stack_alphabet_size, _ = layer_args
                if args.nd_actions in ('sum', 'normalize'):
                    push, repl, pop = logits_to_actions(raw_actions, num_states, stack_alphabet_size, False)
                    actions = torch.stack([
                        push.sum(dim=(1, 2, 3, 4)),
                        repl.sum(dim=(1, 2, 3, 4)),
                        pop.sum(dim=(1, 2, 3))
                    ], dim=1)
                    if args.nd_actions == 'normalize':
                        actions = torch.softmax(actions, dim=1)
                else:
                    actions = raw_actions
            else:
                raise ValueError

        self.run(args, stack_attention_type, stack_attention_args, input_string_strs, actions)

def read_input_tokens(path):
    with path.open() as fin:
        return fin.readline().split()
