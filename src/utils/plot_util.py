import contextlib
import pathlib
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tikzplotlib

def add_plot_arguments(parser, outputs=(None,)):
    for prefix in outputs:
        add_prefixed_plot_arguments(parser, prefix)
    parser.add_argument('--show', action='store_true', default=False,
        help='Show the plot using Tk.')
    parser.add_argument('--legend-columns', type=int)

def add_prefixed_plot_arguments(parser, prefix):
    arg_prefix = '' if prefix is None else f'{prefix}-'
    def add_argument(name, **kwargs):
        parser.add_argument(f'--{arg_prefix}{name}', **kwargs)
    add_argument('title',
        help='Title of the figure.')
    add_argument('width', type=float, default=4.5,
        help='Width of the figure.')
    add_argument('height', type=float, default=3.5,
        help='Height of the figure.')
    add_argument('output', type=pathlib.Path, action='append', default=[],
        help='Output file. Format is controlled by the file extension.')
    add_argument('pgfplots-output', type=pathlib.Path,
        help='PGFPlots output file for LaTeX.')
    add_argument('pgfplots-strict-mode', action='store_true', default=False)
    add_argument('separate-legend', action='store_true', default=False,
        help='Do not show the legend in the main figure.')
    add_argument('legend-width', type=float, default=6.5)
    add_argument('legend-height', type=float, default=0.7)
    add_argument('legend-output', type=pathlib.Path, action='append', default=[],
        help='Output file for the legend.')
    add_argument('legend-pgfplots-output', type=pathlib.Path)

@contextlib.contextmanager
def run_plot(args, outputs=(None,)):
    figures_and_axes = init_plot(args, outputs=outputs)
    if len(figures_and_axes) == 1:
        yield figures_and_axes[0]
    else:
        yield figures_and_axes
    finish_plot(args, figures_and_axes, outputs=outputs)

def init_plot(args, outputs):
    plt.rcParams.update({
        'font.family' : 'serif',
        'text.usetex' : False,
        'pgf.rcfonts' : False
    })
    figures_and_axes = []
    for name in outputs:
        fig, ax = plt.subplots()
        figures_and_axes.append((fig, ax))
        fig.set_size_inches(get_arg(args, name, 'width'), get_arg(args, name, 'height'))
        title = get_arg(args, name, 'title')
        if title is not None:
            ax.set_title(title)
    return figures_and_axes

def get_arg(args, prefix, name):
    key = name if prefix is None else f'{prefix}_{name}'
    return getattr(args, key)

def finish_plot(args, figures_and_axes, outputs):
    for (fig, ax), name in zip(figures_and_axes, outputs):
        if not get_arg(args, name, 'separate_legend'):
            ax.legend(ncols=args.legend_columns)
        plt.figure(fig.number)
        plt.tight_layout()
    if args.show:
        plt.show()
    for (fig, ax), name in zip(figures_and_axes, outputs):
        for output_path in get_arg(args, name, 'output'):
            fig.savefig(output_path)
        pgfplots_output = get_arg(args, name, 'pgfplots_output')
        if pgfplots_output is not None:
            tikzplotlib.save(pgfplots_output, figure=fig, strict=get_arg(args, name, 'pgfplots_strict_mode'))
        legend_output = get_arg(args, name, 'legend_output')
        legend_pgfplots_output = get_arg(args, name, 'legend_pgfplots_output')
        if legend_output or legend_pgfplots_output:
            if legend_output:
                write_legend(args, name, ax)
            if legend_pgfplots_output is not None:
                write_legend_tex(args, name, fig, ax)

def write_legend(args, name, ax):
    legend_fig = plt.figure()
    legend_fig.set_size_inches(get_arg(args, name, 'legend_width'), get_arg(args, name, 'legend_height'))
    legend_ax = legend_fig.add_subplot(111)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_ax.legend(legend_handles, legend_labels, loc='center', frameon=False, ncol=len(legend_labels))
    legend_ax.axis('off')
    for output_path in get_arg(args, name, 'legend_output'):
        legend_fig.savefig(output_path)

def write_legend_tex(args, name, fig, ax):
    if get_arg(args, name, 'separate_legend'):
        ax.legend()
    tikz_code = tikzplotlib.get_tikz_code(figure=fig, strict=get_arg(args, name, 'pgfplots_strict_mode'))
    colors, entries = get_legend_tex_parts(tikz_code)
    legend_pgfplots_output = get_arg(args, name, 'legend_pgfplots_output')
    legend_pgfplots_output.write_text(get_legend_tex(args, name, colors, entries))

def get_legend_tex(args, name, colors, entries):
    if len(colors) != len(entries):
        raise ValueError
    parts = []
    parts.append('\\begin{tikzpicture}\n')
    for definecolor, color in colors:
        parts.append(definecolor)
        parts.append('\n')
    legend_height = get_arg(args, name, 'legend_height')
    legend_columns = -1 if args.legend_columns is None else args.legend_columns
    parts.append(f'''\
\\begin{{axis}}[
  hide axis,
  height={legend_height}in,
  legend style={{
    draw=none,
    /tikz/every even column/.append style={{column sep=0.4cm}}
  }},
  legend columns={legend_columns},
  xmin=0,
  xmax=1,
  ymin=0,
  ymax=1
]
''')
    for (definecolor, color), entry in zip(colors, entries):
        parts.append(f'\\addlegendimage{{{color}}}\n')
        parts.append(entry)
        parts.append('\n')
    parts.append('\\end{axis}\n')
    parts.append('\\end{tikzpicture}\n')
    return ''.join(parts)

LEGEND_TEX_RE = re.compile(r'^(\\definecolor\{([^}]*)\}.*\}|(\\addlegendentry\{.*\}))$', re.M)

def get_legend_tex_parts(code):
    colors = []
    entries = []
    for m in LEGEND_TEX_RE.finditer(code):
        line, color, entry = m.groups()
        if color is not None:
            colors.append((line, color))
        elif entry is not None:
            entries.append(entry)
    return colors, entries

def force_integer_ticks(x_or_y_axis):
    x_or_y_axis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5]))

def get_markers(n):
    """Given an integer n, return n distinct line markers. The markers are
    simple lines drawn at different angles."""
    # The angles need to be in the range [0, 180).
    # We need to avoid 90 degrees, since this is not visible in the legend.
    # To do this, when n is even, divide 180 by n+1 instead of n.
    steps = n + 1 if n % 2 == 0 else n
    for i in range(n):
        angle = i * 180 / steps
        yield (2, 0, angle)
