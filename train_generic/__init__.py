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

from .datautil import dataset_batch, dataset_compact, dataset_enumerate, is_dataset_batched
from .datautil import dataset_flatten, dataset_onehot, dataset_prefetch, dataset_repeat
from .datautil import dataset_set_shapes, dataset_shuffle, dataset_unbatch

import numpy as np


def permutation(x):
    """Return the indices of random permutation of `x`"""
    return np.random.permutation(len(x) if hasattr(x, '__len__') else int(x))
    
    
def mnist(return_type='tensorflow',
          subsample=False,
          take_n=3000,
          take_split=0.8,
          shuffle=True,
          vectorize=True,
          batch_size=None,
          buffer_size=60000,
          val_split=0.1,
          return_val_set=False,
          return_test=True,
          DEFAULT_TRAIN_SIZE=60000):
    assert return_type in ['tensorflow', 'numpy']
    import tensorflow as tf

    (x_train, y_train), (x_test , y_test) = tf.keras.datasets.mnist.load_data()

    if return_type=='tensorflow':
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_test  = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        vec_func = lambda x,y: (
            tf.reshape(x, [x.shape[0]*x.shape[1]]), y 
        )
        normalize_img = lambda img, lbl: (tf.cast(img, tf.float32) / 255., lbl)

        val_take_n = int((take_n if subsample else DEFAULT_TRAIN_SIZE) * val_split)
        buffer_size = take_n if subsample else buffer_size
        test_take = int((1-take_split)*buffer_size)+1

        # Train
        ds_train = ds_train.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.shuffle(buffer_size) if shuffle else ds_train
        ds_train = ds_train.take(take_n) if subsample else ds_train
        ds_train = ds_train.map(vec_func) if vectorize else ds_train
        
        # Test
        ds_test = ds_test.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(vec_func) if vectorize else ds_test
        ds_test = ds_test.shuffle(test_take).take(test_take) if subsample else ds_test
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        if return_val_set:
            ds_val   = ds_train.take(val_take_n).cache()
            ds_train = ds_train.skip(val_take_n).cache()
            ds_test  = ds_test.cache()

            ds_train = ds_train.batch(batch_size) if batch_size is not None else ds_train
            ds_val   = ds_val.batch(batch_size)   if batch_size is not None else ds_val
            ds_test  = ds_test.batch(batch_size)  if batch_size is not None else ds_test

            ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
            ds_val   = ds_val.prefetch(tf.data.experimental.AUTOTUNE)
            ds_test  = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
            return ds_train, ds_val, ds_test

        ds_train = ds_train.batch(batch_size) if batch_size is not None else ds_train
        ds_test  = ds_test.batch(batch_size)  if batch_size is not None else ds_test

        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        ds_test  = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        return_val = (ds_train, ds_test) if return_test else ds_train
        for x in ds_train: break
        print("Loading Tesnorflow dataset with shape: {}".format(x[0].shape))
        return return_val

    if shuffle:
        train_idx = permutation(len(x_train))
        test_idx  = permutation(len(x_test))

        x_train = x_train[train_idx]
        y_train = y_train[train_idx]
        x_test  = x_test[test_idx]
        y_test  = y_test[test_idx]

    if subsample:
        x_train = x_train[:take_n]
        y_train = y_train[:int(take_n*take_split)]

    if vectorize:
        x_train = np.reshape(
            x_train, [x_train.shape[0], x_train.shape[1]*x_train.shape[2]]
        )
        x_test  = np.reshape(
            x_test, [x_test.shape[0], x_test.shape[1]*x_test.shape[2]]
        )

    # Normalize images
    x_train = x_train.astype(float) / 255.0
    x_test  = x_test.astype(float) / 255.0

    if return_val_set:
        perm = permutation(len(x_train))
        val_take_n = int(len(x_train) * val_split)
        x_val = x_train[perm[:val_take_n]]
        y_val = y_train[perm[:val_take_n]]
        x_train = x_train[perm[val_take_n+1:]]
        y_train = y_train[perm[val_take_n+1:]]
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    return_val = ((x_train, y_train), (x_test, y_test)) \
        if return_test else (x_train, y_train)

    print_string = "Loading numpy MNIST with shape:\ntrain: {}\ntest:  {}".format(
        x_train.shape, x_test.shape) \
            if return_test else "Loading numpy MNIST with shape:\ntrain: {}".format(
                x_train.shape
            )
    print(print_string)
    return return_val



