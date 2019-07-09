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

    p = go.Position(board=position.copy(), komi=0, ko=ko, to_play=to_play)
    is_legal = p.all_legal_moves()
    # minigo/go all_legal_moves returns a list of 1 or 0 for legal or not.
    return np.extract(is_legal == 1, np.arange(len(is_legal)))


def apply(action, to_play, board):
    # Pass
    if action is board.size:
        return board.copy()
    p = go.Position(board=board.copy(), to_play=to_play)
    q = p.play_move(coords.from_flat(action))
    if q.ko is not None:
        q.board[q.ko] = 4
    return q.board.copy()
