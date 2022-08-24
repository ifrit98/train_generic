#!/bin/bash

tmux new -d -s fr_train
tmux send-keys -t fr_train.0 "python -c 'from fr_train import training_run, FLAGS; training_run(FLAGS)'" ENTER
tmux a -t fr_train