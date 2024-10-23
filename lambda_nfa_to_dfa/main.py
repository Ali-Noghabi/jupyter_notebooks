
import networkx as nx
import matplotlib.pyplot as plt


class NFA:
    def __init__(self, states, alphabet, accept_states, transitions, start_state):
        self.states = set(states)
        self.alphabet = set(alphabet)
        self.accept_states = set(accept_states)
        self.transitions = transitions
        self.start_state = start_state

    def epsilon_closure(self, state):
        if isinstance(state, int):
            arr = [state]
        elif isinstance(state, set):
            arr = list(state)
        else:
            raise TypeError("State must be an int or a set of ints")

        closure = set(arr)
        for s in arr:
            if s in self.transitions and '$' in self.transitions[s]:
                for next_state in self.transitions[s]['$']:
                    if next_state not in closure:
                        closure.add(next_state)
                        closure |= self.epsilon_closure(next_state)
        return closure


def nfa_to_dfa(nfa):
    dfa_states = set()
    dfa_accept_states = set()
    dfa_transitions = {}
    dfa_start_state = frozenset(nfa.epsilon_closure(nfa.start_state))
    unmarked_states = [dfa_start_state]

    while unmarked_states:
        current_state = unmarked_states.pop()
        dfa_states.add(current_state)

        for symbol in nfa.alphabet:
            next_state = frozenset().union(*[nfa.epsilon_closure(next_state)
                                             for next_state in [nfa.transitions.get(state, {}).get(symbol, set())
                                                                for state in current_state]])
            if not next_state:
                continue
            if next_state not in dfa_states:
                unmarked_states.append(next_state)

            dfa_transitions.setdefault(current_state, {})[symbol] = next_state

        if any(state in nfa.accept_states for state in current_state):
            dfa_accept_states.add(current_state)
            
    dfa_states = {frozenset(state) for state in dfa_states}
    dfa_accept_states = {frozenset(state) for state in dfa_accept_states}
    dfa_transitions = {frozenset(start_state): {symbol: frozenset(end_state) for symbol, end_state in symbols.items()}
                       for start_state, symbols in dfa_transitions.items()}

    return DFA(dfa_states, nfa.alphabet, dfa_accept_states, dfa_transitions, dfa_start_state)


def simulate_dfa(dfa, input_str):
    current_state = dfa.start_state
    for symbol in input_str:
        if symbol not in dfa.alphabet:
            return False
        if symbol not in dfa.transitions[current_state]:
            # Symbol not defined for current state
            return False
        current_state = dfa.transitions[current_state][symbol]
    return current_state in dfa.accept_states


class DFA:
    def __init__(self, states, alphabet, accept_states, transitions, start_state):
        self.states = set(states)
        self.alphabet = set(alphabet)
        self.accept_states = set(accept_states)
        self.transitions = transitions
        self.start_state = start_state


def read_input():
    num_states, num_symbols, num_accept_states, num_transitions, num_strings = map(
        int, input().split())
    alphabet = [input().strip() for _ in range(num_symbols)]
    start_state = int(input().strip())
    accept_states = set(map(int, input().split()))
    transitions = {}
    for _ in range(num_transitions):
        start, symbol, end = input().strip().split()
        transitions.setdefault(int(start), {}).setdefault(
            symbol, set()).add(int(end))
    strings = []
    for _ in range(num_strings):
        strings.append(input())
    return NFA(range(num_states), alphabet, accept_states, transitions, start_state), strings


nfa, strings = read_input()
dfa = nfa_to_dfa(nfa)

for element in strings:
    if simulate_dfa(dfa , element):
        print("Yes")
    else:
        print("No")
# print("res", simulate_dfa(dfa, 'aab'))
# print("res", simulate_dfa(dfa, 'aaaab'))
# print("res", simulate_dfa(dfa, 'b'))
# print("res", simulate_dfa(dfa, 'aabb'))
# print("res", simulate_dfa(dfa, 'aa'))
# print("res", simulate_dfa(dfa, 'aaa'))
# print("res", simulate_dfa(dfa, 'aaba'))
G = nx.DiGraph()

for start_state, symbols in dfa.transitions.items():
    for symbol, end_state in symbols.items():
        node1 = str(start_state)[10:-1]
        edge = str(symbol)
        node2 = str(end_state)[10:-1]
        node1 = node1[1:-1]
        node2 = node2[1:-1]

        print(node1, "--", edge, "->", node2)
        if node1 != node2:  # If it's not a self-loop edge
            G.add_edge(node1, node2, label=edge)
        else:  # If it's a self-loop edge
            G.add_edge(node1, node2, label=edge, selfloop=True)

# Draw the graph
pos = nx.random_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=500)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels)
plt.axis('off')
plt.show()
