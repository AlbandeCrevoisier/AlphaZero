from numpy import array, zeros, ones, moveaxis

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, add
from tensorflow.keras.losses import mean_squared_error
from tensorflow.nn import softmax_cross_entropy_with_logits
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD


def get_input_features(config, history, index):
    """ Takes the 16 last moves up until index included. """
    if index == -1:
        to_play = -1 if len(history) % 2 == 0 else 1
    else:
        to_play = -1 if len(history[:index + 1]) % 2 == 0 else 1
    goban_size = config['goban_size']
    moves = history[max(0, index - 15) : index + 1]
    while len(moves) < 16:
        moves.insert(0, zeros((goban_size, goban_size)))
    moves.append(to_play * ones((goban_size, goban_size)))
    # Move channel last
    moves = moveaxis(moves, 0, -1)
    return moves


def model(config):
    """AlphaZero network model.

    The defaults are the values from the paper.
    - input
        - 8 last moves by each player
        - a layer indicating whose turn it is
    - body
        - single convolutional block
        - nresiduals residual blocks
    - two heads
        - policy: logits of probability for each move, including pass
        - value: scalar
    """
    goban_size = config['goban_size']
    nfilters = config['nfilters']
    nresiduals = config['nresiduals']
    c = config['l2_param']
    
    input = Input((goban_size, goban_size, 17))

    # First block
    x = Conv2D(nfilters, 3, 1, 'same', kernel_regularizer=l2(c))(input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Residual blocks
    for _ in range(nresiduals):
        tmp = Conv2D(nfilters, 3, 1, 'same', kernel_regularizer=l2(c))(x)
        tmp = BatchNormalization()(tmp)
        tmp = LeakyReLU()(tmp)
        tmp = Conv2D(nfilters, 3, 1, 'same',kernel_regularizer=l2(c))(x)
        tmp = BatchNormalization()(tmp)
        x = add([x, tmp])
        x = LeakyReLU()(x)

    # Policy head, outputs logits
    p = Conv2D(2, 1, 1, 'same', kernel_regularizer=l2(c))(x)
    p = BatchNormalization()(p)
    p = LeakyReLU()(p)
    p = Flatten()(p)
    p = Dense(goban_size * goban_size + 1)(p)

    # Value head
    v = Conv2D(1, 1, 1, 'same', kernel_regularizer=l2(c))(x)
    v = BatchNormalization()(v)
    v = LeakyReLU()(v)
    v = Flatten()(v)
    v = Dense(nfilters)(v)
    v = LeakyReLU()(v)
    v = Dense(1, activation='tanh')(v)

    return Model(inputs=input, outputs=[p, v])


def loss(y_true, y_pred):
    l = mean_squared_error(y_true[1], y_pred[1])
    l += softmax_cross_entropy_with_logits(y_true[0], y_pred[0])
    return l


def compile(m: Model):
    m.compile(loss=loss, optimizer=SGD(2e-2, 0.9))
