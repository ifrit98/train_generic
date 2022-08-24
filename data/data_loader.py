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

