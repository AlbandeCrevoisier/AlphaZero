from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, add
from tensorflow.keras.losses import mean_squared_error
from tensorflow.nn import softmax_cross_entropy_with_logits_v2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD


def model(config):
    """AlphaZero network model..

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
    batch_size = config['batch_size']
    goban_size = config['goban_size']
    nfilters = config['nfilters']
    nresiduals = config['nresiduals']
    c = config['l2_param']
    
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

    # Policy head, outputs logits
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


def compile(m: Model)
    return m.compile(loss=loss, optimizer=SGD(2e-2, 0.9))
