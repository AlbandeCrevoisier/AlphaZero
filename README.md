# Reimplementation of DeepMind's Alpha Zero using keras/tensorflow.

The project has been downscaled as follows:
- 5x5 goban
- smaller network & exploration - see config.py
- monothread algorithm

#### Files:
model.py - model of the neural net
mcts.py - Monte-Carlo Tree Search
go_wrapper.py - go engine wrapper
go.py & coords.py - minigo's go engine
config.py - configuration file
pseudocode.py - DeepMind's pseudocode
modified_pseudocode.py - adaptation of DeepMind's pseudocode to my resources.

DeepMind's pseudocode will eventually be removed.
Please note that this is a beginner's work learning keras/tensorflow, NN, & Alpha Zero's
algorithm, do not expect particularly clean code - yet.

#### Roadmap:
- score using chinese rules
- detect the end of the game
- score the game
- make batches
- add network training
- alternate self play & network training
