from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, add


def model(goban_size=19, nfilters=256, nresiduals=19):
    """AlphaZero model.

    The defaults are the values from the paper.
    Architecture:
    - body
        - single convolutional block
        - nresiduals residual blocks
    - two heads
        - policy
        - value
    """
    x = Input((goban_size, goban_size, 17))

    # First block
    x = Conv2D(nfilters, (3, 3), (1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Residual blocks
    for _ in range(nresiduals):
        tmp = Conv2D(nfilters, (3, 3), (1, 1))(x)
        tmp = BatchNormalization()(tmp)
        tmp = LeakyReLU()(tmp)
        tmp = Conv2D(nfilters, (3, 3), (1, 1))(tmp)
        tmp = BatchNormalization()(tmp)
        x = add(x, tmp)
        x = LeakyReLU()(x)

    # Policy head
    p = Conv2D(2, (1, 1), (1, 1))(x)
    p = BatchNormalization()(p)
    p = LeakyReLU()(p)
    p = Flatten()(p)
    p = Dense(goban_size * goban_size + 1)(p)

    # Value head
    v = Conv2D(1, (1, 1), (1, 1))(x)
    v = BatchNormalization()(v)
    v = LeakyReLU()(v)
    v = Flatten()(v)
    v = Dense(nfilters)(v)
    v = LeakyReLU()(v)
    v = Dense(1, activation='tanh')(v)

    return Model(inputs=x, outputs=[p, v])
