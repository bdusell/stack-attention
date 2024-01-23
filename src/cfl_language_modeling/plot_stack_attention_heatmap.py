from cfl_language_modeling.plot_stack_attention_heatmap_util import PlotStackAttentionHeatmap
from cfl_language_modeling.cfl_stack_attention_util import CFLAdapter

class Program(PlotStackAttentionHeatmap, CFLAdapter):
    pass

Program().main()
