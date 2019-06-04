from mcts import *
from model import *
from config import *

c = config
h = []
m = model(c)
compile(m)
m.predict()
a, r = mcts(c, h, m)
