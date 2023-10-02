from ..formal_models.cfg import Rule
from .pdt import Transition, EPSILON

def print_transducer_as_dot(transducer, fout):
    fout.write('digraph {\n')
    fout.write('\tnode [shape=circle];\n')
    states = sorted(transducer.states)
    state_ids = { s : i for i, s in enumerate(states) }
    fout.write('\tqstart [label="",shape=none];\n')
    for state in states:
        if transducer.is_accept_state(state):
            shape = 'doublecircle'
        else:
            shape = 'circle'
        fout.write('\tq%d [label="%s",shape=%s];\n' % (state_ids[state], dot_label(state), shape))
    fout.write('\tqstart -> q%d;\n' % state_ids[transducer.start_state])
    for transition in sorted(transducer.transitions):
        fout.write('\tq%d -> q%d [label="%s"];\n' % (
            state_ids[transition.state_from],
            state_ids[transition.state_to],
            transition_dot_label(transition))
        )
    fout.write('}')

def dot_label(obj):
    return str(obj)

EPSILON_DOT_LABEL = '&epsilon;'

def rule_dot_label(rule):
    return '{%s &rarr; %s}' % (rule.left, string_dot_label(rule.right))

def object_dot_label(obj):
    if obj == EPSILON:
        return EPSILON_DOT_LABEL
    elif isinstance(obj, Rule):
        return rule_dot_label(obj)
    elif isinstance(obj, Transition):
        return transition_dot_label(obj)
    else:
        return dot_label(obj)

def string_dot_label(obj_list):
    if obj_list:
        return ''.join(map(object_dot_label, obj_list))
    else:
        return EPSILON_DOT_LABEL

def transition_dot_label(transition):
    return '%s, %s &rarr; %s : %s' % (
        object_dot_label(transition.input_symbol),
        object_dot_label(transition.popped_symbol),
        object_dot_label(transition.pushed_symbol),
        string_dot_label(transition.output_string)
    )

def print_parse_tree_as_dot(tree, fout):
    fout.write('digraph {\n')
    nodes = list(tree.preorder_traversal())
    nodes_by_id = { n : i for i, n in enumerate(nodes) }
    for i, n in enumerate(nodes):
        fout.write('\tn%d [label="%s"];\n' % (i, n.value))
    for i, n in enumerate(nodes):
        for c in n.children:
            fout.write('\tn%d -> n%d;\n' % (i, nodes_by_id[c]))
    fout.write('}')
