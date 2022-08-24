import os
import time
from xml.dom import NotFoundErr
import yaml
from pprint import pformat
import matplotlib.pyplot as plt

from .train_wrapper import train
from .stream_logger import file_logger
from ._startup import inside_docker, import_file, import_flags

try:
    GLOBAL_FLAGS = import_flags('flags.yaml')
except:
    GLOBAL_FLAGS = {}

#Defaults for legible figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams["image.cmap"] = 'jet'

timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))


HPARAM_DEFAULTS = {
    "epochs": 10,
    "steps_per_epoch": 100,
    "units": 128,
    "batch_size": 32,
    "init_learning_rate": 1e-3,
    "max_learning_rate": 1,
    "min_learning_rate": 1e-6,
    "buffer_size": 100,
    "verbose": False
}

_globals = {
    'runs_dir': GLOBAL_FLAGS.get('runs_dir', None),
    'run_dir': {
        'path': None,
        'FLAGS': GLOBAL_FLAGS if GLOBAL_FLAGS != {} else None
    },
    'eval_metrics': None
}

HISTORY = None
LOGGER = None


def clear_run():
    """
    Clears `_globals` dictionary by setting all to `None`
    """
    _globals['runs_dir'] = None
    _globals['run_dir']['path'] = None
    _globals['run_dir']['FLAGS'] = None

def unique_run_dir(runs_dir = None, format_="%m_%d_%y_%H-%M-%S"):
    """Returns a unique run directory filepath to house an experiment"""
    runs_dir = runs_dir or _globals['runs_dir']
    run_dir = time.strftime(format_, time.strptime(time.asctime()))
    return os.path.join(runs_dir, run_dir)

def is_run_active():
    return _globals['run_dir']['path'] is None

def get_run_dir():
    return _globals['run_dir']['path'] if is_run_active() else os.getcwd()


def do_training_run(run_dir, meta_file='metadata.json'):
    """
    Perform the training run with current run_dir. Sets cwd to run_dir, and then 
    executes training.  Logs created by redirecting stdout to `logfile`.

    Clears cache and returns to original working directory before returning.

    Args: 
        run_dir: String path to current run directory.
        meta_file: json filepath to metadata output file for dumping `globals`. 
          Defaults to `metadata.json`
    
    Returns:
        None
    """
    global LOGGER

    _globals['start_time'] = timestamp()
    LOGGER.info('Start time: {}'.format(_globals['start_time']))
    LOGGER.info("Using run directory: {}".format(run_dir))
    owd = os.getcwd()
    os.chdir(run_dir)

    LOGGER.info("Executing training...")
    FLAGS = _globals['run_dir'].get('FLAGS', None)

    # GET MODEL_FN by loading
    model_cfg = FLAGS.get('model_cfg', 'models/model.mpy')
    if isinstance(model_cfg, str):
        model_path = model_cfg
    else:
        model_path = model_cfg.get('model_src_file')
    model_module = import_file(model_path)
    model_fn = model_module.__dict__.get(model_cfg.get('model_fn_name', 'build_model'))
    if model_fn is None:
        raise NotFoundErr("Must provide proper str name for model module function.")

    # Get data loader by fn name
    data_cfg = FLAGS.get('data_cfg', {})
    if data_cfg == {}: raise NotFoundErr("Must provide `data_cfg` dict.")

    data_fp = data_cfg.get('data_loader_file', None)
    data_fn = data_cfg.get('data_loader_fn_name', None)

    if data_fn is None: raise NotFoundErr("Must provide `data_loader_fn_name` str.")
    data_module = import_file(data_fp)
    data_fn = data_module.__dict__.get(data_fn)

    train_cfg = FLAGS.get('train_cfg', {})
    if train_cfg == {}: raise NotFoundErr("Must provide `train_cfg` dict.")

    print("Calling train()... writing to {}".format(_globals['run_dir']['path']))
    history, _ = train(
        model_fn=model_fn,
        model_cfg=model_cfg,
        data_fn=data_fn,
        data_cfg=data_cfg,
        **train_cfg,
    )

    LOGGER.info("History:\n{}".format(pformat(history)))

    # Record end time
    _globals['end_time'] = timestamp()
    LOGGER.info('End time: {}'.format(_globals['end_time']))

    # Save and log results
    import json
    with open(meta_file, 'w') as f:
      json.dump(_globals,  f)

    LOGGER.info('_globals\n: {}'.format(pformat(_globals)))

    clear_run()
    os.chdir(owd)


def initialize_run(run_dir=None,
                   logger_name='init_log',
                   FLAGS=None):
    """
    Initializes training run variables.

    Args:
        run_dir: String path to current run directory.
        flags: flags object, as a python dictionary from yaml file.

    """
    # if _globals['runs_dir'] is None:
    #     _globals['runs_dir'] = os.path.abspath('/training/data/runs')
    
    print("Runs_dir:", _globals['runs_dir'])
    if not os.path.exists(_globals['runs_dir']):
        os.mkdir(_globals['runs_dir'])

    if run_dir is None:
        run_dir = unique_run_dir()
    print("run_dir:", run_dir)

    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    global LOGGER
    LOGGER = file_logger(os.path.join(run_dir, logger_name))
    if FLAGS is not None:
        LOGGER.info("FLAGS:\n{}".format(pformat(FLAGS)))

        # Copy over flags for evaluation
        with open(os.path.join(run_dir, 'flags.yaml'), 'w') as f:
            yaml.dump(FLAGS, f, sort_keys=False)

    # Copy over eval script for convenience
    cmd = "cp ./eval_trained_model.sh " + os.path.join(run_dir, 'eval_trained_model.sh')
    if os.path.exists('./eval_trained_model.sh'):
        os.system(cmd)

    cmd = "cp ./eval_model_ckpt.sh " + os.path.join(run_dir, 'eval_model_ckpt.sh')
    if os.path.exists('./eval_model_ckpt.sh'):
        os.system(cmd)

    _globals['run_dir']['path'] = run_dir
    _globals['run_dir']['FLAGS'] = FLAGS

    LOGGER.info("Initialized run_dir {}".format(run_dir))
    LOGGER.info("Globals:\n{}".format(pformat(_globals)))

    return run_dir


def training_run(FLAGS=None, run_dir=None, runs_dir=None, 
                 default_docker_path='/traiing/data/runs',
                 default_non_docker_path=os.path.expanduser('~/runs')):
    """Initialize and perform a training run given `file_` training script.
    Args:
        run_dir -- the unique run directory to place experiment metadata
        runs_dir -- high level runs directory that houses all runs. Defaults to `~/runs`
        FLAGS -- FLAGS object or dictionary if already loaded
    """
    global LOGGER

    clear_run()

    using_docker = inside_docker()

    if FLAGS is None:
        FLAGS = GLOBAL_FLAGS

    # FLAGS runs_dir has highest priority.
    if FLAGS is not None and FLAGS.get('runs_dir', False):
        runs_dir = FLAGS['runs_dir'] #metadir (not timestamped subdir)
    elif runs_dir is not None:
        runs_dir = os.path.abspath(default_non_docker_path)
    else:
        runs_dir = None

    runs_dir = default_docker_path \
        if using_docker else (default_non_docker_path if runs_dir is None else runs_dir)

    _globals['runs_dir'] = runs_dir
    # print("\nglobals['runs_dir']", _globals['runs_dir'])

    # print("\nrun_dir in training_run()", run_dir)
    run_dir = initialize_run(run_dir=run_dir, FLAGS=FLAGS)

    do_training_run(run_dir)
    
    LOGGER.info("Training run completed: {}".format(run_dir))
    raise ValueError("0 <- Training completed successfully.  IGNORE VALUE ERROR")

