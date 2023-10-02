import collections
import itertools

from ..util import group_by
from ..formal_models.cfg import Grammar, Rule, Nonterminal, Terminal
from .pdt import EPSILON

class TransducerIndex:
    """
    A data structure which makes queries on the pushdown transducer in Lang's
    algorithm efficient.
    """

    def __init__(self, transducer):
        self.transducer = transducer
        self.scanning_pop = group_by(
            (
                t for t in transducer.transitions
                if t.input_symbol != EPSILON and t.popped_symbol != EPSILON
            ),
            key=lambda t: (t.state_from, t.popped_symbol, t.input_symbol)
        )
        self.scanning_nopop = group_by(
            (
                t for t in transducer.transitions
                if t.input_symbol != EPSILON and t.popped_symbol == EPSILON
            ),
            key=lambda t: (t.state_from, t.input_symbol)
        )
        self.non_scanning_pop = group_by(
            (
                t for t in transducer.transitions
                if t.input_symbol == EPSILON and t.popped_symbol != EPSILON
            ),
            key=lambda t: (t.state_from, t.popped_symbol)
        )
        self.non_scanning_nopop = group_by(
            (
                t for t in transducer.transitions
                if t.input_symbol == EPSILON and t.popped_symbol == EPSILON
            ),
            key=lambda t: t.state_from
        )

    def scanning_transitions(self, state, stack_top, symbol):
        return itertools.chain(
            self.scanning_pop.get((state, stack_top, symbol), []),
            self.scanning_nopop.get((state, symbol), [])
        )

    def non_scanning_transitions(self, state, stack_top):
        return itertools.chain(
            self.non_scanning_pop.get((state, stack_top), []),
            self.non_scanning_nopop.get(state, [])
        )

class StackBottomType:

    def __str__(self):
        return '$'

    def __repr__(self):
        return 'STACK_BOTTOM'

STACK_BOTTOM = StackBottomType()

class StartSymbolType:

    def __str__(self):
        return 'S'

    def __repr__(self):
        return 'S'

def simulate_transducer(transducer_index, string):
    """
    Run Lang's algorithm on a string of symbols.

    Parameters
    ----------
    transducer_index : TransducerIndex
    string : iterable
        An iterable of objects. The objects may be of any type that supports
        equality and hashing.

    Returns
    -------
    project.formal_models.cfg.Grammar
        A context-free grammar whose language is equal to the set of all
        strings produced by the transducer when run on the input string.
    """
    step = initial_step(transducer_index)
    for xi in string:
        step = step.simulate_non_scanning_transitions()
        step = step.simulate_scanning_transitions(xi)
    step = step.simulate_non_scanning_transitions()
    return step.construct_grammar()

def initial_step(transducer_index):
    """
    Return an object that represents the initial configuration of Lang's
    algorithm.

    Returns
    -------
    PreNonScanningStep
    """
    q0 = transducer_index.transducer.start_state
    U0_mode = (q0, STACK_BOTTOM, 0)
    U0 = (U0_mode, U0_mode)
    U0_subset = []
    U0_record = (U0, U0_subset)
    U0_subset.append(U0_record)
    Li = [U0_record]
    Ai = { a_key(U0) : U0_record }
    Ei = collections.defaultdict(list, { e_key(U0_mode) : U0_subset })
    P = [Rule(Nonterminal(U0), ())]
    P_set = set(P)
    return PreNonScanningStep(transducer_index, 0, Li, Ai, Ei, P, P_set)

def simulate_non_scanning_transitions(transducer_index, i, Li, Ai, Ei, P, P_set):
    Wi = []

    def add_item(V, right_side, V_subset):
        key = a_key(V)
        V_record = Ai.get(key)
        if V_record is None:
            V_record = (V, V_subset)
            Ai[key] = V_record
            Ei[e_key(V[0])].append(V_record)
            Li.append(V_record)
            # Perform deferred pop operations
            for W_record, t in Wi:
                W, W_subset = W_record
                if W[1] == V[0]:
                    # Y = V was needed when processing U = W
                    process_pop(W_record, V_record, t)
        rule = Rule(Nonterminal(V), right_side)
        if rule not in P_set:
            P.append(rule)
            P_set.add(rule)

    def process_pop(U_record, Y_record, t):
        r = t.state_to
        C = t.pushed_symbol
        z = t.output_string
        U, U_subset = U_record
        ((p, A, _), (q, B, j)) = U
        assert _ == i
        Y, Y_subset = Y_record
        (_, (s, D, k)) = Y
        assert _ == (q, B, j)
        V = ((r, B, i), (s, D, k))
        right_side = [Nonterminal(Y), Nonterminal(U)] + as_terminals(z)
        V_subset = Y_subset
        add_item(V, right_side, V_subset)

    index = 0
    while index < len(Li):
        U, U_subset = U_record = Li[index]
        ((p, A, _), (q, B, j)) = U
        assert _ == i
        for t in transducer_index.non_scanning_transitions(p, A):
            r = t.state_to
            C = t.pushed_symbol
            z = t.output_string
            if t.popped_symbol == EPSILON:
                if C == EPSILON:
                    # Ignores the stack
                    V = ((r, A, i), (q, B, j))
                    right_side = [Nonterminal(U)] + as_terminals(z)
                    V_subset = U_subset
                    add_item(V, right_side, V_subset)
                else:
                    # Pushes to the stack
                    V = ((r, C, i), (p, A, i))
                    right_side = as_terminals(z)
                    V_subset = Ei[e_key(V[1])]
                    add_item(V, right_side, V_subset)
            else:
                if C == EPSILON:
                    # Pops from the stack
                    index2 = 0
                    while index2 < len(U_subset):
                        Y_record = U_subset[index2]
                        # If a new record is added to Y_subset as a result
                        # of this operation, it will be processed later in
                        # this loop.
                        process_pop(U_record, Y_record, t)
                        index2 += 1
                    if j == i:
                        # If U was needed by U in the previous loop, then
                        # the left and right sides of U are the same. If
                        # both sides are the same, then U was added to
                        # U_subset at some point before the final iteration
                        # of the loop, so the pair (U, U) has already been
                        # processed. So, the deferred computations for U
                        # should be added to Wi *after* the above loop, or
                        # else (U, U) will be processed twice.
                        Wi.append((U_record, t))
                else:
                    # Replaces top of stack
                    V = ((r, C, i), (q, B, j))
                    right_side = [Nonterminal(U)] + as_terminals(z)
                    V_subset = U_subset
                    add_item(V, right_side, V_subset)
        index += 1

def simulate_scanning_transitions(transducer_index, xi, i, Li, Ei, P, P_set):
    xi = Terminal(xi)
    Li_next = []
    Ai = {}
    prev_Ei = Ei
    Ei = collections.defaultdict(list)

    def add_item(V, right_side, V_subset):
        key = a_key(V)
        V_record = Ai.get(key)
        if V_record is None:
            V_record = (V, V_subset)
            Ai[key] = V_record
            Ei[e_key(V[0])].append(V_record)
            Li_next.append(V_record)
        P.append(Rule(Nonterminal(V), right_side))

    for U_record in Li:
        U, U_subset = U_record
        ((p, A, _), (q, B, j)) = U
        assert _ == i
        for t in transducer_index.scanning_transitions(p, A, xi):
            r = t.state_to
            C = t.pushed_symbol
            z = t.output_string
            if t.popped_symbol == EPSILON:
                if C == EPSILON:
                    # Ignores the stack
                    V = ((r, A, i + 1), (q, B, j))
                    right_side = [Nonterminal(U)] + as_terminals(z)
                    V_subset = U_subset
                    add_item(V, right_side, V_subset)
                else:
                    # Pushes to the stack
                    V = ((r, C, i + 1), (p, A, i))
                    right_side = as_terminals(z)
                    V_subset = prev_Ei[e_key(V[1])]
                    add_item(V, right_side, V_subset)
            else:
                if C == EPSILON:
                    # Pops from the stack
                    for Y, Y_subset in U_subset:
                        (_, (s, D, k)) = Y
                        assert _ == (q, B, j)
                        V = ((r, B, i + 1), (s, D, k))
                        right_side = [Nonterminal(Y), Nonterminal(U)] + as_terminals(z)
                        V_subset = Y_subset
                        add_item(V, right_side, V_subset)
                else:
                    # Replaces top of stack
                    V = ((r, C, i + 1), (q, B, j))
                    right_side = [Nonterminal(U)] + as_terminals(z)
                    V_subset = U_subset
                    add_item(V, right_side, V_subset)
    Li = Li_next
    return Li, Ai, Ei

def construct_grammar(transducer_index, Li, P):
    # NOTE This modifies the list P in-place.
    q0 = transducer_index.transducer.start_state
    S = Nonterminal(StartSymbolType())
    for U_record in Li:
        U, U_subset = U_record
        ((r, A, _), (q, B, j)) = U
        if (
            transducer_index.transducer.is_accept_state(r) and
            A == STACK_BOTTOM and
            q == q0 and
            B == STACK_BOTTOM and
            j == 0
        ):
            P.append(Rule(S, (Nonterminal(U),)))
    return Grammar(start=S, rules=P)

def a_key(U):
    ((p, A, i), (q, B, j)) = U
    return (p, A, q, B, j)

def e_key(mode):
    (p, A, i) = mode
    return (p, A)

def as_terminals(z):
    return [Terminal(zz) for zz in z]

class Step:
    pass

class PreNonScanningStep(Step):
    """
    Represents a configuration of Lang's algorithm just prior to simulating
    non-scanning transitions.
    """

    def __init__(self, transducer_index, i, Li, Ai, Ei, P, P_set):
        self.transducer_index = transducer_index
        self.i = i
        self.Li = Li
        self.Ai = Ai
        self.Ei = Ei
        self.P = P
        self.P_set = P_set

    def simulate_non_scanning_transitions(self):
        """
        Simulate all of the non-scanning transitions in the PDT at this stage
        of the algorithm.

        Returns
        -------
        PreScanningStep
        """
        simulate_non_scanning_transitions(
            self.transducer_index, self.i, self.Li, self.Ai, self.Ei, self.P,
            self.P_set)
        return PreScanningStep(
            self.transducer_index, self.i, self.Li, self.Ei, self.P,
            self.P_set)

class PreScanningStep(Step):
    """
    Represents a configuration of Lang's algorithm just prior to scanning an
    input symbol.
    """

    def __init__(self, transducer_index, i, Li, Ei, P, P_set):
        self.transducer_index = transducer_index
        self.i = i
        self.Li = Li
        self.Ei = Ei
        self.P = P
        self.P_set = P_set

    def simulate_scanning_transitions(self, xi):
        """
        Scan a symbol and simulate all of the appropriate scanning transitions.

        Returns
        -------
        PreNonScanningStep
        """
        Li, Ai, Ei = simulate_scanning_transitions(
            self.transducer_index, xi, self.i, self.Li, self.Ei, self.P,
            self.P_set)
        return PreNonScanningStep(
            self.transducer_index, self.i + 1, Li, Ai, Ei, self.P, self.P_set)

    def construct_grammar(self):
        """
        Construct the CFG that represents the set of all strings generated by
        the PDT up to this point.

        Returns
        -------
        project.formal_models.cfg.Grammar
        """
        # TODO It would be nice to avoid this linear copy.
        P = list(self.P)
        return construct_grammar(self.transducer_index, self.Li, P)
