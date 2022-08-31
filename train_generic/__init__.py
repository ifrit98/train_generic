from ._startup import *

set_based_gpu()

try:
    FLAGS = import_flags('./flags.yaml')
except:
    from warnings import warn
    warn("Could not load `flags.yaml`...")

from .eval import evaluate_model

from .train_wrapper import train, train_lite

from .set_gpu import set_based_gpu

from .run_manager import training_run

from . import utils
from . import models

from .data import data_loader_mnist

from .datautil import dataset_batch, dataset_compact, dataset_enumerate, is_dataset_batched
from .datautil import dataset_flatten, dataset_onehot, dataset_prefetch, dataset_repeat
from .datautil import dataset_set_shapes, dataset_shuffle, dataset_unbatch

from .model_paces import model_paces, paces_demo
