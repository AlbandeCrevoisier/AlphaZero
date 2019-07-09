#!/usr/bin/python
import numpy as np
from mcts import *
from model import *
from config import *
from go import *
from go_wrapper import *

c = config
h = [np.zeros((5, 5))]
m = model(c)
compile(m)
to_play = 1

for _ in range(50):
    a, _ = mcts(c, h.copy(), m)
    h.append(apply(a, to_play, h[-1]))
    print(type(h[-1]))
    to_play *= -1

print(h[-1])
print(Position(h[-1]).score())
