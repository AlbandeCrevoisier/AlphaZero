import numpy as np
import go, coords


def terminal(self):
    pass


def terminal_value(self, to_play):
    pass


def legal_actions(position, prev_position):
    """ Return all legal moves as flattened coordinates. """
    # Without ko yet
    p = go.Position(board=position.copy())
    return p.all_legal_moves()


def apply(action, position):
    return go.Position(board=position).play_move(coords.from_flat(action)).board
