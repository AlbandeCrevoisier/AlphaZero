import numpy as np
from mcts import *
from model import *
from config import *
from go import *
from go_wrapper import *

h = [np.zeros((5, 5))]
child_visits = []
img = []
target_policies = []
target_values = []
m = model(config)
compile(m)
to_play = 1

for i in range(100):
    action, root = mcts(config, h, m)
    apply(action, h)
    sum_visits = sum(c.nvisits for c in root.children.values())
    visits = [root.children[a].nvisits / sum_visits if a in root.children else 0
        for a in range(26)]
    child_visits.append(visits)
    # A komi of 3.5 forces B to look for the best possible play.
    # http://www.mathpuzzle.com/go.html
    target_policies.append(visits)
    target_values.append(Position(h[-1], komi=3.5).result())
    img.append(get_input_features(config, h, -1))

m.fit(np.array(img), [target_policies, target_values], epochs=config['nsteps'])
