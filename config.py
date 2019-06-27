config = {
    # Model                      Original Paper values
    'batch_size': 3,        # 4096
    'goban_size': 5,          # 19
    'nfilters': 3,           # 256
    'nresiduals': 3,          # 19
    'l2_param': 1e-4,          # 1e-4
    # MCTS
    'c_base': 19652,           # 19652
    'c_init': 1.25,            # 1.25
    'nsamples': 3,            # 30
    'dirichlet': 0.03,         # 0.03
    'explo_fraction': 0.25,    # 0.25
    'nsim': 10                # 800
    }
