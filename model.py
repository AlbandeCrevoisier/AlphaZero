from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, add
from tensorflow.keras.losses import mean_squared_error
from tensorflow.nn import softmax_cross_entropy_with_logits_v2
from tensorflow.keras.regularizers import l2


def model(goban_size=19, nfilters=256, nresiduals=19, c=1e-4):
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
    input = Input((goban_size, goban_size, 17))

    # First block
    x = Conv2D(nfilters, 3, padding='same', kernel_regularizer=l2(c))(input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Residual blocks
    for _ in range(nresiduals):
        tmp = Conv2D(nfilters, 3, padding='same', kernel_regularizer=l2(c))(x)
        tmp = BatchNormalization()(tmp)
        tmp = LeakyReLU()(tmp)
        tmp = Conv2D(nfilters, 3, padding='same', kernel_regularizer=l2(c))(x)
        tmp = BatchNormalization()(tmp)
        x = add([x, tmp])
        x = LeakyReLU()(x)

    # Policy head
    p = Conv2D(2, 1, padding='same', kernel_regularizer=l2(c))(x)
    p = BatchNormalization()(p)
    p = LeakyReLU()(p)
    p = Flatten()(p)
    p = Dense(goban_size * goban_size + 1)(p)

    # Value head
    v = Conv2D(1, 1, padding='same', kernel_regularizer=l2(c))(x)
    v = BatchNormalization()(v)
    v = LeakyReLU()(v)
    v = Flatten()(v)
    v = Dense(nfilters)(v)
    v = LeakyReLU()(v)
    v = Dense(1, activation='tanh')(v)

    return Model(inputs=input, outputs=[p, v])


def loss(y_true, y_pred):
    l = mean_squared_error(y_true[1], y_pred[1])
    l += softmax_cross_entropy_with_logits_v2(y_true[0], y_pred[0])
    return l
