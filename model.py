from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, add


def model(goban_size=19, num_filters=256, num_residuals=19):
    """AlphaZero model.

    The defaults are the values from the paper.
    Architecture:
    - body
        - single convolutional block
        - num_residuals residual blocks
    - two heads
        - policy
        - value
    """
    input = Input((goban_size, goban_size, 17))

    # First block
    x = Conv2D(num_filters, (3, 3), (1, 1), padding='same')(input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Residual blocks
    for _ in range(num_residuals):
        tmp = Conv2D(num_filters, (3, 3), (1, 1), padding='same')(x)
        tmp = BatchNormalization()(tmp)
        tmp = LeakyReLU()(tmp)
        tmp = Conv2D(num_filters, (3, 3), (1, 1), padding='same')(x)
        tmp = BatchNormalization()(tmp)
        x = add([x, tmp])
        x = LeakyReLU()(x)

    # Policy head
    p = Conv2D(2, (1, 1), (1, 1), padding='same')(x)
    p = BatchNormalization()(p)
    p = LeakyReLU()(p)
    p = Flatten()(p)
    p = Dense(goban_size * goban_size + 1)(p)

    # Value head
    v = Conv2D(1, (1, 1), (1, 1), padding='same')(x)
    v = BatchNormalization()(v)
    v = LeakyReLU()(v)
    v = Flatten()(v)
    v = Dense(num_filters)(v)
    v = LeakyReLU()(v)
    v = Dense(1, activation='tanh')(v)

    return Model(inputs=input, outputs=[p, v])
