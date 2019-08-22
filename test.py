import numpy as np
from mcts import *
from model import *
from config import *
from go import *
from go_wrapper import *

c = config
h = [np.zeros((5, 5))]
cvisits = [[]]
m = model(c)
compile(m)
to_play = 1

for _ in range(50):
    a, r = mcts(c, h, m)
    apply(a, h)
    svisits = sum(c.nvisits for c in r.children.values())
    v = [r.children[a].nvisits / svisits if a in r.children else 0
        for a in range(26)]
    print(v)
    cvisits.append(v)

print(h[-1])
print("Score: ", Position(h[-1], komi=0).score())

print(cvisits[-1])
print(len(cvisits), len(h))
