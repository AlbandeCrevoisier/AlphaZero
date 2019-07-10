"""Monte Carlo Tree Search: variant of the PUCT"""
from math import exp, log, sqrt
from numpy import array
from numpy.random import gamma
from model import get_input_features
from go_wrapper import legal_actions, apply


class Node:

    def __init__(self, prior: float):
        self.nvisits = 0
        self.tot_val = 0
        self.prior = prior
        self.children = {} # {action: child}

    def value(self):
        if self.nvisits == 0:
            return 0
        return self.tot_val / self.nvisits


def make_children(node: Node, policy_logits, history):
    position = history[-1]
    policy = {a: exp(policy_logits[a]) for a in legal_actions(history)}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


def add_exploration_noise(config, node: Node):
    actions = node.children.keys()
    noise = gamma(config['dirichlet'], 1, len(actions))
    frac = config['explo_fraction']
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def back_propag(search_path, value: float):
    is_player_turn = True
    for node in search_path:
        node.tot_val += value if is_player_turn else (1 - value)
        is_player_turn = not is_player_turn
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


def select_action(config, root: Node, nmoves):
    nvisits = [(child.nvisits, action)
               for action, child in root.children.items()]
    if nmoves < config['maxmoves']:
        # TODO: use softmax sampling to force diversity.
        # _, action = softmax_sample(nvisits)
        _, action = max(nvisits)
    else:
        _, action = max(nvisits)
    return action


def mcts(config, history, model):
    root = Node(0)
    image = [[get_input_features(config, history, -1)]]
    policy_logits, _ = model.predict(image)
    make_children(root, policy_logits[0], history)
    add_exploration_noise(config, root)

    for _ in range(config['nsim']):
        node = root
        tmp_history = history.copy()
        search_path = [root]

        while len(node.children) != 0:
            action, node = select_child(config, node)
            apply(action, tmp_history)
            search_path.append(node)

        to_play = -1 if len(tmp_history) % 2 == 0 else 1
        policy_logits, value = model.predict(
            [[get_input_features(config, tmp_history, -1)]])
        make_children(node, policy_logits[0], tmp_history)
        back_propag(search_path, value)
    return select_action(config, root, len(history)), root
