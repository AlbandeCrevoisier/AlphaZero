"""Monte Carlo Tree Search: variant of the PUCT"""
from math import log, sqrt


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


def mcts(config):
    pass


def select_action(config, node: Node):
    _, action, child = max((uc_score(config, node, child), action, child)
                           for action, child in node.children.items())
    return action, child


def uc_score(config, parent: Node, child: Node):
    c_base, c_init = config['c_base'], config['c_init']
    c = log((1 + parent.nvisits + c_base) / c_base) + c_init
    u = c * child.prior * sqrt(parent.nvisits) / (1 + child.nvisits)
    return child.value() + u
