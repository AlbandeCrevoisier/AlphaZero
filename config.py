config = {
    # Model
    'batch_size': 4096,
    'goban_size': 19,
    'nfilters': 256,
    'nresiduals': 19,
    'l2_param': 1e-4,
    #MCTS
    'c_base': 19652,
    'c_init': 1.25,
    'nsamples': 30,
    'dirichlet': 0.03,
    'explo_fraction': 0.25,
    'nsim': 800
    }
