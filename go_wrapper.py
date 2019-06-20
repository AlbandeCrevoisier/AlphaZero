import numpy as np
import go, coords


def terminal(self):
    pass


def terminal_value(self, to_play):
    pass


def legal_actions(position, prev_position):
    """ Return all legal moves as flattened coordinates. """
    p = go.Position(board=position.copy())
    moves = p.all_legal_moves()
    # brut force ko finding
    for m in moves:
        if (apply(m, position) == prev_position).all() == True:
            moves.remove(m)
            break
    return moves


def apply(action, position):
    return go.Position(board=position).play_move(coords.from_flat(action)).board
