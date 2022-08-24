import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import BatchDataset

from .utils import is_scalar, as_integer_tensor, is_tensor, as_tensor, is_scalar_tensor


def alleq(x):
    try:
        iter(x)
    except:
        x = [x]
    current = x[0]
    for v in x:
        if v != current:
            return False
    return True     


def compact(x, x_keys, y_keys):
    """Compacts the `x` dictionary into (x, y), potentially nested
       tuple pairs for use with `tf.keras.Model.fit()`
    """
    x_scalar = is_scalar(x_keys)
    y_scalar = is_scalar(y_keys)
    if x_scalar:
        if y_scalar:
            return (x[x_keys], x[y_keys])
        else:
            return (x[x_keys], tuple(x[k] for k in y_keys))
    else:
        if y_scalar:
            return (tuple(x[k] for k in x_keys), x[y_keys])
    return (tuple(x[k] for k in x_keys), tuple(x[k] for k in y_keys))

# Forces tf.data.Dataset object to be keras.Model.fit() compatible: An (x, y) tuple.
def dataset_compact(ds, compact_x_key='signal', compact_y_key='target', num_parallel_calls=4):
    """Condense dataset object down to contain only (x, y) tuple pairs.

    Args:
        ds: A dataset (tf.data.Dataset) object

        compact_x_key: string or iterable of keys for `x` (input) tensors.

        compact_y_key: string or iterable of keys for `y` (output) tensors.

    pair = (x['signal'], x['target'])
    
    This is to be called immedately preceeding passing the dataset to 
    `tf.keras.Model.fit()`:
        import tensorflow as tf
        from tf_dataset import *
        
        input = tf.keras.Input(shape=[1024], dtype='float32')
        output = tf.keras.layers.Dense(input, activation='softmax')
        model = tf.keras.Model(input, ouput)
        model.compile(...)

        data_dir = "./data"
        df = construct_metadata(data_dir)
        ds = signal_dataset(df)
        ds = dataset_compact(ds, x_key='signal', y_key='target')

        h = model.fit(ds, ...)
    """
    def compact(x):
        nonlocal compact_x_key, compact_y_key
        x_scalar = is_scalar(compact_x_key)
        y_scalar = is_scalar(compact_y_key)

        if x_scalar:
            if y_scalar:
                return (x[compact_x_key], x[compact_y_key])
            else:
                return (x[compact_x_key], tuple(x[k] for k in compact_y_key))
        else:
            if y_scalar:
                return (tuple(x[k] for k in compact_x_key), x[compact_y_key])
            else:
                return (tuple(x[k] for k in compact_x_key), tuple(x[k] for k in compact_y_key))

    return ds.map(
        compact, 
        num_parallel_calls=as_tensor(num_parallel_calls, 'int64'))

# TODO: Make sure whis works both nested and not
# TODO: fix load_AIS data to allow batching?? (`data`: tf.string)?
def dataset_set_shapes(ds):
    for nb in ds: break
    shapes = {
        k: v.shape.as_list() if type(v) is not dict \
                             else {
                                 _k: _v.shape.as_list() for _k,_v in v.items()
                            } for k,v in nb.items()}
    del nb
    def set_shapes(x):
        [v.set_shape(shapes[k]) if type(shapes[k]) is not dict \
                                else {ki: vi.set_shape(shapes[k][ki]) for ki,vi in v.items()}\
                                for k,v in x.items()]
        return x
    return ds.map(set_shapes)

def dataset_onehot(ds, target_key='target'):
    def onehot(x):
        x[target_key] = tf.one_hot(
            as_integer_tensor(x[target_key]), 
            as_integer_tensor(x['num_classes']),
            dtype=tf.int64
        )
        return x
    return ds.map(onehot)

def dataset_flatten(ds, key):
    def flatten(x):
        x[key] = tf.keras.layers.Flatten()(x[key])
        return x
    return ds.map(flatten)

def dataset_batch(ds, batch_size):
    batch_size = as_tensor(batch_size, 'int64')
    if not is_scalar_tensor(batch_size):
        raise ValueError("`batch_size` must be a scalar.")
    return ds.batch(batch_size, drop_remainder=True)

def dataset_shuffle(ds, shuffle_buffer_size, reshuffle_each_iteration=True):
    shuffle_buffer_size = as_tensor(shuffle_buffer_size, 'int64')
    if not is_scalar_tensor(shuffle_buffer_size):
        raise ValueError("`shuffle_buffer_size` must be a scalar.")
    return ds.shuffle(
        shuffle_buffer_size, reshuffle_each_iteration=reshuffle_each_iteration)

def dataset_repeat(ds, count=None):
    return ds.repeat(count=count)

def dataset_unbatch(ds):
    return ds.unbatch()

def dataset_prefetch(ds, n_prefetch=1):
    return ds.prefetch(n_prefetch)

def is_dataset_batched(ds):
    if  ds.__class__ == BatchDataset:
        return True
    for x in ds: break
    return False if any(list(map(lambda v: v.ndim == 0 if is_tensor(v) else True, x.values()))) \
         else alleq(list(map(lambda a: tf.shape(a)[0].numpy(), x.values())))

def dataset_enumerate(ds, start=0):
    return ds.enumerate(as_tensor(start, "int64"))


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
          return_test=True):
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

        # Train
        ds_train = ds_train.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.take(take_n) if subsample else ds_train
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(buffer_size) if shuffle else ds_train
        ds_train = ds_train.map(vec_func) if vectorize else ds_train
        ds_train = ds_train.batch(batch_size) if batch_size is not None else ds_train
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        # Test
        ds_test = ds_test.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(vec_func) if vectorize else ds_test
        if subsample:
            ds_train = ds_train.shuffle(int(take_split*buffer_size))
            ds_train = ds_train.take(int(take_n*take_split))
        ds_test = ds_test.batch(batch_size) if batch_size is not None else ds_test
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

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

    return_val = ((x_train, y_train), (x_test, y_test)) \
        if return_test else (x_train, y_train)

    print_string = "Loading numpy MNIST with shape:\ntrain: {}\ntest:  {}".format(
        x_train.shape, x_test.shape) \
            if return_test else "Loading numpy MNIST with shape:\ntrain: {}".format(
                x_train.shape
            )
    print(print_string)
    return return_val

