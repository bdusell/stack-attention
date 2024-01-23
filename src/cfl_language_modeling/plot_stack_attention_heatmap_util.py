import numpy

from utils.plot_util import add_plot_arguments, run_plot
from cfl_language_modeling.analyze_stack_attention_util import AnalyzeStackAttention

class PlotStackAttentionHeatmap(AnalyzeStackAttention):

    def add_arguments(self, parser):
        add_plot_arguments(parser)

    def run(self, args, stack_attention_type, stack_attention_args, input_token_strs, actions):
        data = actions
        if stack_attention_type == 'superposition':
            action_labels = [
                'push',
                'no-op',
                'pop'
            ]
            heatmap_type = 'probabilistic'
        elif stack_attention_type == 'nondeterministic':
            if args.nd_actions in ('sum', 'normalize'):
                action_labels = [
                    'push',
                    'replace',
                    'pop'
                ]
                if args.nd_actions == 'sum':
                    heatmap_type = 'diverging'
                else:
                    heatmap_type = 'sequential'
            else:
                num_states, stack_alphabet_size, _ = stack_attention_args
                action_labels = get_nondeterministic_stack_action_labels(
                    num_states=num_states,
                    stack_alphabet_size=stack_alphabet_size
                )
                heatmap_type = 'diverging'
        with run_plot(args) as (fig, ax):
            sequence_length, num_actions = data.size()
            if len(input_token_strs) + 1 != sequence_length:
                raise ValueError
            if len(action_labels) != num_actions:
                raise ValueError

            ax.set_xlabel('Action')
            ax.set_ylabel('$\\leftarrow$ Symbol')

            left = 0
            top = 0
            bottom, right = data.size()
            extent = (left, right, bottom, top)

            if heatmap_type == 'probabilistic':
                color_options = dict(
                    cmap='Greys',
                    vmin=0.0,
                    vmax=1.0
                )
            elif heatmap_type == 'sequential':
                color_options = dict(
                    cmap='Greys',
                )
            elif heatmap_type == 'diverging':
                biggest_abs = data.abs().max()
                color_options = dict(
                    cmap='bwr',
                    vmin=-biggest_abs,
                    vmax=biggest_abs
                )
            else:
                raise ValueError
            im = ax.imshow(
                data.cpu().numpy(),
                aspect='auto',
                interpolation='none',
                extent=extent,
                **color_options
            )
            cbar = ax.figure.colorbar(im, ax=ax)
            if heatmap_type == 'probabilistic':
                cbarlabel = 'Probability'
            else:
                cbarlabel = 'Log-Weight'
            cbar.ax.set_ylabel(cbarlabel)
            # Add the y axis labels in between rows.
            bos = 'BOS'
            eos = 'EOS'
            input_labels = [bos, *input_token_strs, eos]
            ax.set_yticks(range(len(input_labels)), labels=input_labels)
            ax.tick_params(axis='y', labelsize=8)
            # Remove the tick lines from the x axis.
            ax.tick_params(
                axis='x',
                length=0,
                labelsize=8
            )
            # Add the x axis labels in the middle of columns.
            ax.set_xticks(
                numpy.arange(num_actions) + 0.5,
                labels=action_labels,
                ha='right',
                rotation=45,
                rotation_mode='anchor'
            )
            # Remove the black border.
            for k in ('top', 'right', 'bottom', 'left'):
                ax.spines[k].set_visible(False)

def get_nondeterministic_stack_action_labels(num_states, stack_alphabet_size):
    labels = []
    for q in range(num_states):
        q_str = f'q_{{{q}}}'
        for x in range(stack_alphabet_size):
            x_str = stack_symbol_to_str(x)
            push_labels = []
            repl_labels = []
            pop_labels = []
            for r in range(num_states):
                r_str = f'q_{{{r}}}'
                for y in range(stack_alphabet_size):
                    y_str = stack_symbol_to_str(y)
                    push_labels.append(f'push ${q_str}, {x_str} \\rightarrow {r_str}, {y_str}$')
                    repl_labels.append(f'repl ${q_str}, {x_str} \\rightarrow {r_str}, {y_str}$')
                pop_labels.append(f'pop ${q_str}, {x_str} \\rightarrow {r_str}$')
            labels.extend(push_labels)
            labels.extend(repl_labels)
            labels.extend(pop_labels)
    return labels

def stack_symbol_to_str(y):
    if y == 0:
        return '\\bot'
    else:
        return f'\\mathtt{{{y-1}}}'
