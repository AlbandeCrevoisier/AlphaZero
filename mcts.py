"""Monte Carlo Tree Search: variant of the PUCT"""
from go import legal_actions
from math import exp, log, sqrt


class Node:

    def __init__(self, prior: float):
        self.nvisits = 0
        self.tot_val = 0
        self.prior = prior
        self.children = {} #action: child

    def value(self):
        if self.nvisits == 0:
            return 0
        return self.tot_val / self.nvisits


def make_children(node: Node, policy_logits, position):
    policy = {a: exp(policy_logits[a]) for a in legal_actions(position)}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


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


