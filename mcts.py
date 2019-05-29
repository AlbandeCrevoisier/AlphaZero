"""Monte Carlo Tree Search: variant of the PUCT"""
from math import exp, log, sqrt
from numpy.random import gamma
from go import legal_actions


class Node:

    def __init__(self, prior: float, to_play):
        self.nvisits = 0
        self.to_play = -1 # 1 for Black, -1 for White.
        self.tot_val = 0
        self.prior = prior
        self.children = {} # {action: child}

    def value(self):
        if self.nvisits == 0:
            return 0
        return self.tot_val / self.nvisits


def make_children(node: Node, policy_logits, position):
    policy = {a: exp(policy_logits[a]) for a in legal_actions(position)}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum, -node.to_play)


def add_exploration_noise(config, node: Node):
    actions = node.children.keys()
    noise = gamma(config['dirichlet'], 1, len(actions))
    frac = config['explo_fraction']
    for a, n in zip(actions, noise):
        node.children[a].prior *= (1 - frac) + n * frac


def back_propag(search_path: List[Node], value: float, to_play):
    for node in search_path:
        node.tot_val += value if node.to_play == to_play else (1 - value)
        node.nvisits += 1


def ucb_score(config, parent: Node, child: Node):
    c_base, c_init = config['c_base'], config['c_init']
    c = log((1 + parent.nvisits + c_base) / c_base) + c_init
    u = c * child.prior * sqrt(parent.nvisits) / (1 + child.nvisits)
    return child.value() + u


def select_child(config, node: Node):
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in node.children.items())
    return action, child


def select_action(config, root: Node, history):
    nvisits = [(child.nvisits, action)
               for action, child in root.children.items()]
    if len(history) < config['nsamples']:
        _, action = softmax_sample(nvisits)
    else:
        _, action = max(nvisits)
    return action


def mcts(config, history, model):
    root = Node(0)
    _, policy_logits = model.predict() #TODO add param with history
    make_children(root, policy_logits, history)
    add_exploration_noise(config, root)

    for _ in range(config.nsim):
        node = root
        h = history
        search_path = [node]

        while len(node.children) != 0:
            action, node = select_child(config, node)
            scratch_game.apply(action) # TODO apply turn to temp_history
            search_path.append(node)

        value, policy_logits = model.predict() #TODO add param with temp_history
        make_children(node, policy_logits, temp_history)
        back_propag(search_path, value, scratch_game.to_play()) # TODO scrath_game.to_play
    return select_action(config, game, root), root
