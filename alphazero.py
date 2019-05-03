""" Reimplementation of Alpha Zero."""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, add

goban_size = 9
nfilters = 256
nresiduals = 2 # 19 in the original paper.

Input(goban_size, goban_size, 17)

# Residual tower: a block followed by 19 residual blocks.
Conv2D(nfilters, (3, 3), (1, 1))
BatchNormalization()
LeakyReLU()

Conv2D(nfilters, (3, 3), (1, 1))
BatchNormalization()
LeakyReLU()
Conv2D(nfilters, (3, 3), (1, 1))
BatchNormalization()
add(input, intermediate_value)
LeakyReLU()

# Two heads.
# Policy head
Conv2D(2, (1, 1), (1, 1))
BatchNormalization()
LeakyReLU()
Flatten()
Dense(goban_size * goban_size + 1)

# Value head
Conv2D(1, (1, 1), (1, 1))
BatchNormalization()
LeakyReLU()
Flatten()
Dense(nfilters)
LeakyReLU()
Dense(1, activation='tanh')
