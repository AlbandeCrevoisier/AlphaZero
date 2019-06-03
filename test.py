from mcts import *
from model import *
from config import *

c = config
h = []
m = model(c)
compile(m)
a, r = mcts(c, h, m)

