""" Go engine wrapper
position: N*N numpy array.
action: (x, y) where the origin is the top left.
ko: an action.
to_play: 1 for Black, -1 for White.

As of now, ko are handled as follows:
manually get Position.ko and write it as '4' on the goban.
This is clunky but is the path of least resistance to interface
minigo/go.Position with the model.
"""


import numpy as np
import go, coords


def terminal(self):
    pass


def terminal_value(self, to_play):
    pass


def legal_actions(to_play, position):
    """ Return all legal moves as flattened coordinates. """
    # Extract ko
    t = np.where(position == 4)
    if t[0].size is 0:
        ko = None
    else:
        # Is there a better way of unwrapping this?
        ko = (t[0][0], t[1][0])
    position[ko] = 0

    p = go.Position(board=board, ko=ko)
    return p.all_legal_moves()


def apply(action, to_play, board):
    p = go.Position(board=board)
    q = p.play_move(coords.from_flat(action), color=to_play)
    if q.ko is not None:
        q.board[q.ko] = 4
    return q.board.copy()
