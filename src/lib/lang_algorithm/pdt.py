class EpsilonType:

    def __str__(self):
        return 'ε'

    def __repr__(self):
        return 'EPSILON'

EPSILON = EpsilonType()

class Transducer:

    def __init__(self, start_state, transitions, accept_states):
        if not isinstance(transitions, list):
            transitions = list(transitions)
        if not isinstance(accept_states, set):
            accept_states = set(accept_states)
        states = set()
        states.add(start_state)
        for t in transitions:
            states.add(t.state_from)
            states.add(t.state_to)
        states.update(accept_states)
        self.start_state = start_state
        self.transitions = transitions
        self.accept_states = accept_states
        self.states = states

    def is_accept_state(self, state):
        return state in self.accept_states

    def __str__(self):
        return '''\
start state: %s
accept states: %s
transitions:
%s''' % (
            self.start_state,
            self.accept_states,
            '\n'.join(map(str, self.transitions))
        )

    def __repr__(self):
        return 'Transducer(%r, %r, %r)' % (
            self.start_state, self.transitions, self.accept_states)

class Transition:

    def __init__(self, state_from, state_to, popped_symbol=EPSILON,
            input_symbol=EPSILON, pushed_symbol=EPSILON, output_string=None):
        if output_string is None:
            output_string = ()
        elif not isinstance(output_string, tuple):
            output_string = tuple(output_string)
        self.state_from = state_from
        self.state_to = state_to
        self.popped_symbol = popped_symbol
        self.input_symbol = input_symbol
        self.pushed_symbol = pushed_symbol
        self.output_string = output_string

    @property
    def is_scanning(self):
        return self.input_symbol != EPSILON

    def key(self):
        return (
            self.state_from, self.state_to, self.popped_symbol,
            self.input_symbol, self.pushed_symbol, self.output_string
        )

    def __eq__(self, other):
        return type(self) == type(other) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __lt__(self, other):
        if type(self) != type(other):
            raise TypeError
        return self.key() < other.key()

    def __str__(self):
        return '(%s, %s, %s) -> (%s, %s, %s)' % (
            self.state_from, self.input_symbol, self.popped_symbol,
            self.state_to, self.output_string, self.pushed_symbol)

    def __repr__(self):
        return (
            'Transition(state_from=%r, state_to=%r, popped_symbol=%r, '
            'input_symbol=%r, pushed_symbol=%r, output_string=%r)'
        ) % self.key()
