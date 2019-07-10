#!/usr/bin/python
import numpy as np
from go import *
from go_wrapper import *

g = np.array(
[[0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [1, 1, -1, -1, 1],
 [0, 1, -1, 1, 1]])
p = Position(board=g.copy(), to_play=-1)
print(p.all_legal_moves()[:-1].reshape((5,5)))
