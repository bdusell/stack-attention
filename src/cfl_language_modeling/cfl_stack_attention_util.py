from cfl_language_modeling.analyze_stack_attention_util import AnalyzeStackAttention
from cfl_language_modeling.model_util import CFLModelInterface
from cfl_language_modeling.task_util import add_task_arguments, parse_task

class CFLAdapter(AnalyzeStackAttention):

    def get_model_interface_class(self):
        return CFLModelInterface

    def add_vocab_arguments(self, parser):
        add_task_arguments(parser)

    def load_vocab(self, parser, args):
        return parse_task(parser, args)

    def construct_saver(self, model_interface, args, vocabs):
        return model_interface.construct_saver(
            args,
            input_size=vocabs.input_vocab.size(),
            output_size=vocabs.output_vocab.size()
        )

    def get_layers(self, saver):
        return saver.kwargs['transformer_layers']

    def convert_tokens_to_input_tensor(self, model_interface, vocabs, device, input_string_strs):
        input_vocab = vocabs.input_vocab
        str_to_int = {
            input_vocab.value(i) : i
            for i in range(vocabs.input_vocab.size())
        }
        input_string_ints = [str_to_int[s] for s in input_string_strs]
        eos_symbol_int = vocabs.output_vocab.end_symbol
        input_tensor, _ = model_interface.batch_to_tensor_pair(
            [input_string_ints],
            device,
            vocabs.input_vocab.size(),
            eos_symbol_int
        )
        return input_tensor
