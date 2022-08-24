from ._startup import *

set_based_gpu()

FLAGS = import_flags('./flags.yaml')

from .eval import evaluate_model

from .train_wrapper import train

from .set_gpu import set_based_gpu

from .run_manager import training_run

from . import utils

from .datautil import mnist, dataset_batch, dataset_compact, dataset_enumerate
from .datautil import dataset_flatten, dataset_onehot, dataset_prefetch, dataset_repeat
from .datautil import dataset_set_shapes, dataset_shuffle, dataset_unbatch, is_dataset_batched