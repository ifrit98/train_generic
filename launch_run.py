import os
from sys import platform, argv

if platform.__eq__('win32'):
    raise EnvironmentError(
        "\n`mv` and `rsync` commands require a unix-like operating system.\n\n\
        Get on ubuntu or WSL if you need to use pyruns. \n\n\
        Check back later for windows compatibility.")

runs_dir = os.path.expanduser('~/training/runs') if len(argv) == 1 else argv[1]
if not os.path.exists(runs_dir):
    os.mkdir(runs_dir)

if not runs_dir:
    raise ValueError('\nMust supply a runs_dir path.\nE.g. `~/runs` or `C:/runs`\n')

if not os.path.exists(runs_dir):
    print('\nRuns_dir path supplied does not exist. Creating it now...\n')
    try:
        os.mkdir(runs_dir)
    except Exception as e:
        print(e)

print('\n\nStarting training run in top-level run directory: {}'.format(runs_dir.upper()))
from train_generic import training_run, FLAGS
training_run(FLAGS, runs_dir=runs_dir)