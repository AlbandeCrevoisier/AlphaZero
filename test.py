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
    a, _ = mcts(c, h, m)
    apply(a, h)

print(h[-1])
print(Position(h[-1], komi=0).score())
