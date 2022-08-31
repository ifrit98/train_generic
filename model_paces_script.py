import os

from mlcurves import learn_rate_range_test
from mlcurves import complexity_curves_tf
from mlcurves import train_set_size_curves_tf

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.metrics import SparseCategoricalAccuracy, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy



class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __iter__(self):
        return iter(self.__dict__.items())
    def add_to_namespace(self, **kwargs):
        self.__dict__.update(kwargs)

def env(**kwargs):
    return Namespace(**kwargs)
environment = env


############################################################################################
############################################################################################

class Antirectifier(tf.keras.layers.Layer):
  """Build simple custome layer."""

  def __init__(self, initializer="he_normal", **kwargs):
    super(Antirectifier, self).__init__(**kwargs)
    self.initializer = tf.keras.initializers.get(initializer)

  def build(self, input_shape):
    output_dim = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(output_dim * 2, output_dim),
        initializer=self.initializer,
        name="kernel",
        trainable=True,
    )

  def call(self, inputs):  #pylint: disable=arguments-differ
    inputs -= tf.reduce_mean(inputs, axis=-1, keepdims=True)
    pos = tf.nn.relu(inputs)
    neg = tf.nn.relu(-inputs)
    concatenated = tf.concat([pos, neg], axis=-1)
    mixed = tf.matmul(concatenated, self.kernel)
    return mixed

  def get_config(self):
    # Implement get_config to enable serialization. This is optional.
    base_config = super(Antirectifier, self).get_config()
    config = {"initializer": tf.keras.initializers.serialize(self.initializer)}
    return dict(list(base_config.items()) + list(config.items()))


def antirectifier_tiny(input_shape, 
                       num_classes=10, 
                       dense_units=256, 
                       dropout=0.2, 
                       optimizer='rmsprop',
                       from_logits=False):
  if isinstance(optimizer, str): assert optimizer.lower() in ['rmsprop', 'adam']
  model = Sequential(
      [
          Input(shape=input_shape),
          Dense(dense_units),
          Antirectifier(),
          Dense(dense_units),
          Antirectifier(),
          Dropout(dropout),
          Dense(num_classes),
      ]
  )
  if from_logits:
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=RMSprop() if optimizer == 'rmsprop' else Adam(),
        metrics=[SparseCategoricalAccuracy()],
    )
  else:
    model.compile(
      loss=CategoricalCrossentropy(),
      optimizer=RMSprop() if optimizer == 'rmsprop' else Adam(),
      metrics=[CategoricalAccuracy()]
    )
  return model



############################################################################################
############################################################################################



add_time = lambda s: s + '_' + timestamp()


def ploty(y, x=None, xlab='obs', ylab='value', 
          save=True, title='', filepath='plot'):
    sns.set()
    if x is None: x = np.linspace(0, len(y), len(y))
    filepath = add_time(filepath)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel=xlab, ylabel=ylab, title=title)
    ax.grid()
    best_lr = x[np.argmin(y)]
    plt.axvline(x=best_lr, color='r', label='Best LR {:.4f}'.format(best_lr))
    plt.legend()
    if save:
       fig.savefig(filepath)
    plt.show()
    return filepath


def plot_lr_range_test_from_hist(history,
                                filename="lr_range_test",
                                max_loss=5,
                                max_lr=1):
    loss = np.asarray(history.history['loss'])
    lr   = np.asarray(history.history['lr'])
    cut_index = np.argmax(loss > max_loss)
    if cut_index == 0:
        print("\nLoss did not exceed `MAX_LOSS`.")
        print("Increase `epochs` and `MAX_LR`, or decrease `MAX_LOSS`.")
        print("\nPlotting with full history. May be scaled incorrectly...\n\n")
    else:
        loss[cut_index] = max_loss
        loss = loss[:cut_index]
        lr = lr[:cut_index]
    
    lr_cut_index = np.argmax(lr > max_lr)
    if lr_cut_index != 0:
        lr[lr_cut_index] = max_lr
        lr = lr[:lr_cut_index]
        loss = loss[:lr_cut_index]

    ploty(
        loss, lr, 
        xlab='Learning Rate', ylab='Loss', 
        filepath=filename
    )


def infer_best_lr_params(history, factor=3): 
    idx = tf.argmin(history.history['loss'])
    best_run_lr = history.history['lr'][idx]
    min_lr = best_run_lr / factor
    return [min_lr, best_run_lr, idx]


# Reference: https://arxiv.org/pdf/1708.07120.pdf%22
def learn_rate_range_test(model, ds, init_lr=1e-4, factor=3, 
                          plot=True, steps_per_epoch=None,
                          max_lr=3, max_loss=2, epochs=25, 
                          save_hist=True, verbose=1, outpath='lr_range_test'):
    """
    Perform a learn rate range test using a single epoch per learn_rate. (paper version)
    """
    lr_range_callback = tf.keras.callbacks.LearningRateScheduler(
        schedule = lambda epoch: init_lr * tf.pow(
            tf.pow(max_lr / init_lr, 1 / (epochs - 1)), epoch))

    if steps_per_epoch is not None:
        hist = model.fit(
            ds,
            epochs=epochs,
            steps_per_epoch=int(steps_per_epoch),
            callbacks=[lr_range_callback],
            verbose=verbose)
    else:
        hist = model.fit(
            ds,
            epochs=epochs,
            callbacks=[lr_range_callback],
            verbose=verbose)

    if save_hist:
        from pickle import dump
        f = open("lr-range-test-history", 'wb')
        dump(hist.history, f)
        f.close()

    min_lr, best_lr, best_lr_idx = infer_best_lr_params(hist, factor)

    if plot:
        plot_lr_range_test_from_hist(
            hist, 
            max_lr=max_lr, max_loss=max_loss,
            filename=outpath
        )

    return (min_lr, best_lr), hist



############################################################################################
############################################################################################




def process_history(results, histories, acc_nm, loss_nm, val_acc_nm, val_loss_nm):
    train_errs, val_errs, test_errs = [], [], []
    train_losses, train_accs = [], []
    val_losses, val_accs, test_losses, test_accs = [], [], [], []

    for (test_loss, test_acc), history in zip(results.values(), histories.values()):
        val_acc = history.history[val_acc_nm][-1]
        train_acc = history.history[acc_nm][-1]
        train_loss = history.history[loss_nm][-1]
        val_loss = history.history[val_loss_nm][-1]

        train_errs.append(1 - train_acc)
        val_errs.append(1 - val_acc)
        test_errs.append(1 - test_acc)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    total_hist = env()
    total_hist.history = {
        'train_loss': train_losses, 'train_accuracy': train_accs,
        'val_loss': val_losses, 'val_accuracy': val_accs,
        'test_loss': test_losses, 'test_accuracy': test_accs
        }
    return total_hist


from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
import time
timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))
def plot_metrics(history,
                 show=False,
                 save_png=True,
                 xlab=None, ylab=None,
                 xticks=None, xtick_labels=None,
                 outpath='training_curves_{}.pdf'.format(timestamp())):
    sns.set()
    plt.clf()
    plt.cla()

    keys = list(history.history)
    epochs = range(
        min(list(map(lambda x: len(x[1]), history.history.items())))
    ) 

    ax = plt.subplot(211)
    if 'acc' in keys:
        acc = history.history['acc']
    elif 'accuracy' in keys:
        acc = history.history['accuracy']
    elif 'train_acc' in keys:
        acc = history.history['train_acc']
    elif 'train_accuracy' in keys:
        acc = history.history['train_accuracy']
    elif 'sparse_categorical_accuracy' in keys:
        acc = history.history['sparse_categorical_accuracy']
    elif 'categorical_accuracy' in keys:
        acc = history.history['categorical_accuracy']
    else:
        raise ValueError("Training accuracy not found")


    plt.plot(epochs, acc, color='green', 
        marker='+', linestyle='dashed', label='Training accuracy'
    )

    if 'val_acc' in keys:
        val_acc = history.history['val_acc']
        plt.plot(epochs, val_acc, color='blue', 
            marker='o', linestyle='dashed', label='Validation accuracy'
        )
    elif 'val_accuracy' in keys:
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, val_acc, color='blue', 
            marker='o', linestyle='dashed', label='Validation accuracy'
        )
    elif 'val_sparse_categorical_accuracy' in keys:
        val_acc = history.history['val_sparse_categorical_accuracy']
        plt.plot(epochs, val_acc, color='blue', 
            marker='o', linestyle='dashed', label='Validation accuracy'
        )
    elif 'val_categorical_accuracy' in keys:
        val_acc = history.history['val_categorical_accuracy']
        plt.plot(epochs, val_acc, color='blue', 
            marker='o', linestyle='dashed', label='Validation accuracy'
        )
    else:
        warn("Validation accuracy not found... skpping")

    if 'test_acc' in keys:
        test_acc = history.history['test_acc']
        plt.plot(epochs, test_acc, color='red', 
            marker='x', linestyle='dashed', label='Test Accuracy'
        )
    elif 'test_accuracy' in keys:
        test_acc = history.history['test_accuracy']
        plt.plot(epochs, test_acc, color='red', 
            marker='x', linestyle='dashed', label='Test Accuracy'
        )
    else:
        warn("Test accuracy not found... skipping")


    plt.title('Training, validation and test accuracy')
    plt.legend()

    ax2 = plt.subplot(212)
    if 'loss' in keys:
        loss = history.history['loss']
    elif 'train_loss' in keys:
        loss = history.history['train_loss']
    else:
        raise ValueError("Training loss not found")

    plt.plot(epochs, loss, color='green', 
        marker='+', linestyle='dashed', label='Training Loss'
    )

    if 'val_loss' in keys:
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, color='blue', 
            marker='o', linestyle='dashed', label='Validation Loss'
        )    
    elif 'validation_loss' in keys:
        val_loss = history.history['validation_loss']
        plt.plot(epochs, val_loss, color='blue', 
            marker='o', linestyle='dashed', label='Validation Loss'
        )
    else:
        warn("Validation loss not found... skipping")

    if 'test_loss' in keys:
        test_loss = history.history['test_loss']
        plt.plot(epochs, test_loss, color='red', marker='x', label='Test Loss')

    plt.title('Training, validation, and test loss')
    plt.legend()

    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax2.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax2.set_xticklabels(xtick_labels)

    plt.tight_layout()

    if save_png:
        plt.savefig(outpath)
    if show:
        plt.show()



import numpy as np
def train_set_size_curves_tf(model_fn, model_args,
                             train_ds, val_ds, 
                             test_ds, num_classes,
                             batch_size=16,
                             epochs=10, n_runs=11, 
                             shuffle_init=True, 
                             buffer_size=None,
                             outpath='./plot'):
    # Takes model and tensorflow Models and data.Dataset objects ONLY
    # Assume model is compiled properly before being passed as an argument

    # Try unbatching first
    try:
        train_ds = train_ds.unbatch()
        val_ds   = val_ds.unbatch()
        test_ds  = test_ds.unbatch()
    except:
        pass

    # Get monotonically increasing range based on percentages
    train_len = len(list(train_ds.as_numpy_iterator()))

    if not os.path.exists(outpath): os.mkdir(outpath)

    if buffer_size is None:
        buffer_size = train_len

    if shuffle_init:
        train_ds = train_ds.shuffle(buffer_size)
        val_ds = val_ds.shuffle(buffer_size)
        test_ds = test_ds.shuffle(buffer_size)

    u = 1 / n_runs
    rng = np.arange(u, 1+u, u)

    # Split datasets into random subsets of increasing size, e.g. [10%, 20%,..., 100%]
    train_sizes = [int(train_len * p) for p in rng]
    tr_prev = 0

    if model_args.get('input_shape') is None or model_args.get('num_classes') is None:
        for x in train_ds: break
        model_args['input_shape'] = x[0].shape
        model_args['num_classes'] = num_classes
        del x

    histories = {}
    results = {}
    for i, tr in enumerate(train_sizes):
        print("Starting dataset size: (train) {}", tr)
        print("Percentage of full train_ds {}%".format((tr/train_len)*100))

        ds_sub = train_ds.skip(tr_prev).take(tr).shuffle(buffer_size).batch(batch_size)
        tr_prev = tr

        model = model_fn(**model_args)
        history = model.fit(ds_sub, validation_data=val_ds.batch(batch_size), epochs=epochs)
        histories[i] = history

        res = model.evaluate(test_ds.batch(batch_size))
        results[i] = res
        
        plot_metrics(history, show=False, outpath=os.path.join(
            outpath, 'training_curves_{}.pdf'.format(i))
        )

        del model

    # TODO: Ensure this is a consistently deterministic ordering
    hist_names = dict(
        zip(
            ['loss_nm', 'acc_nm', 'val_loss_nm', 'val_acc_nm'], 
            list(history.history)
        )
    )
    total_history = process_history(results, histories, **hist_names)

    plot_metrics(
        total_history, show=False, xlab="Train Set Size (#)", 
        xticks=range(len(train_sizes)), xtick_labels=train_sizes,
        outpath=os.path.join(outpath, 'final_train_size_curves.pdf')
    )
    return total_history



############################################################################################
############################################################################################



from inspect import signature
fargs = lambda f: list(signature(f).parameters.keys())


def get_param_count(model):
    trainableParams = np.sum(
        [np.prod(v.get_shape()) for v in model.trainable_weights]
    )
    nonTrainableParams = np.sum(
        [np.prod(v.get_shape()) for v in model.non_trainable_weights]
    )
    return trainableParams + nonTrainableParams


def complexity_curves_tf(model_fn, 
                         input_shape,
                         num_classes,
                         train_ds, 
                         val_ds, 
                         test_ds, 
                         configs,
                         epochs=10,
                         batch_size=16,
                         outpath='./plot'):

    if not os.path.exists(outpath): os.mkdir(outpath)

    # Try unbatching first
    try:
        train_ds = train_ds.unbatch()
        val_ds   = val_ds.unbatch()
        test_ds  = test_ds.unbatch()
    except:
        pass

    # Must fully prepare data beforehand for model ingestion (e.g. batch, repeat, prefetch)
    train_ds = train_ds.batch(batch_size=batch_size)
    val_ds   = val_ds.batch(batch_size=batch_size)
    test_ds  = test_ds.batch(batch_size=batch_size)

    histories = {}
    results = {}
    model_sizes = {}

    config_by_name = 'model_nm' in fargs(model_fn)

    for i, (nm, cfg) in enumerate(configs.items()):

        if config_by_name:
            model = model_fn(
                input_shape=input_shape, num_classes=num_classes, model_nm=nm
            )
        else:
            cfg.update(dict(input_shape=input_shape, num_classes=num_classes))
            model = model_fn(**cfg)

        h = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=epochs,
            batch_size=batch_size
        )
        histories[i] = h
        plot_metrics(h, show=False, outpath=os.path.join(
            outpath, 'training_curves_{}.pdf'.format(i))
        )

        r = model.evaluate(test_ds)
        results[i] = r

        model_sizes[i] = get_param_count(model)

    model_basename = nm.split("_")[0]
    # TODO: Ensure this is a consistently deterministic ordering
    hist_names = dict(
        zip(
            ['loss_nm', 'acc_nm', 'val_loss_nm', 'val_acc_nm'], 
            list(h.history)
        )
    )
    total_history = process_history(results, histories, **hist_names)

    plot_metrics(
        total_history, show=False, xlab="Model complexity (# params)", 
        ylab="Crossentropy Loss",
        xticks=range(len(model_sizes)), xtick_labels=list(model_sizes.values()),
        outpath=os.path.join(outpath, '{}_complexity_curves.pdf'.format(model_basename))
    )

    return total_history



import os
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


import yaml
def model_paces(model_fn, 
                input_shape, 
                num_classes, 
                train_ds, 
                val_ds,
                test_ds, 
                num_size_runs=11,
                size_curves_epochs=25,
                init_model_key='baseline',
                model_cfg='model_cfg.yaml', # can be a python dict()
                outpath="."):
    """
    Put a model through its paces.

    (1) Runs a learning rate scheduler to find best learning rate parameters for this model, 
    given a particular dataset.

    (2) Runs a routine to train and test model on increasing data set sizes (takes subsets)

    (3) Runs a rounine to infer average best model size (measured in # parameters)

    (4) Returns the results as `dict` and saves plots to `outpath/*.pdf`
    """
    mkdir(outpath)

    if isinstance(model_cfg, str):
        with open(model_cfg, 'rb') as f:
            model_cfg = yaml.load(f)
    
    base_model_args = model_cfg.get(init_model_key, {})
    base_model_args.update(dict(input_shape=input_shape, num_classes=num_classes))

    model = model_fn(**base_model_args) # model_fn(input_shape, num_classes)
    print(model.summary())

    (min_lr, init_lr), h = learn_rate_range_test(
        model, train_ds, outpath=os.path.join(outpath, "lr_range_test")
    )

    train_size_history = train_set_size_curves_tf(
        model_fn, base_model_args, train_ds, val_ds, test_ds, 
        num_classes=num_classes, epochs=size_curves_epochs, n_runs=num_size_runs,
        outpath=os.path.join(outpath, "train_size_test")
    )

    complexity_history = complexity_curves_tf(
        model_fn, input_shape=input_shape, num_classes=num_classes,
        configs=model_cfg, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        outpath=os.path.join(outpath, "complexity_test")
    )

    return {
        'min_lr': min_lr,
        'init_lr': init_lr,
        'train_size_history': train_size_history,
        'complexity_history': complexity_history,
        'lr_range_history': h
    }
    # TODO: Train using pipeline that writes eval out with new `init_lr`




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
          expand_last_dim=False,
          one_hot_labels=False,
          drop_remainder=True,
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

        if expand_last_dim:
            ds_train = ds_train.map(lambda img,lbl: (tf.expand_dims(img, -1), lbl))
            ds_test  = ds_test.map( lambda img,lbl: (tf.expand_dims(img, -1), lbl))

        if one_hot_labels:
            ds_train = ds_train.map(lambda img,lbl: (img, tf.one_hot(lbl, 10, dtype='int32')))
            ds_test  = ds_test.map( lambda img,lbl: (img, tf.one_hot(lbl, 10, dtype='int32')))

        if return_val_set:
            ds_val   = ds_train.take(val_take_n).cache()
            ds_train = ds_train.skip(val_take_n).cache()
            ds_test  = ds_test.cache()

            ds_train = ds_train.batch(
                batch_size, drop_remainder=drop_remainder) if batch_size is not None else ds_train
            ds_val   = ds_val.batch(
                batch_size, drop_remainder=drop_remainder) if batch_size is not None else ds_val
            ds_test  = ds_test.batch(
                batch_size, drop_remainder=drop_remainder) if batch_size is not None else ds_test

            ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
            ds_val   = ds_val.prefetch(tf.data.experimental.AUTOTUNE)
            ds_test  = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
            return ds_train, ds_val, ds_test

        ds_train = ds_train.batch(
            batch_size, drop_remainder=drop_remainder) if batch_size is not None else ds_train
        ds_test  = ds_test.batch(
            batch_size, drop_remainder=drop_remainder) if batch_size is not None else ds_test

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

    if expand_last_dim:
        x_train = np.expand_dims(x_train, -1)
        x_test  = np.expand_dims(x_test, -1)

    if one_hot_labels:
        y_train = tf.one_hot(y_train, 10).numpy().astype(int)
        y_test  = tf.one_hot(y_test, 10).numpy().astype(int)

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



def paces_demo():

    train_ds, val_ds, test_ds = mnist(
        expand_last_dim=False, subsample=True, batch_size=16, 
        one_hot_labels=False,
        return_val_set=True,
        drop_remainder=True
    )

    model_fn=antirectifier_tiny
    input_shape=(784,)
    num_classes=10
    init_model_key='small'
    num_size_runs=11
    size_curves_epochs=25
    outpath='./plots'

    configs = model_cfg = dict(
        tiny=dict(),
        small=dict(
            dense_units=128,
            dropout=0.1,
            from_logits=True
        ),
        baseline=dict(
            dense_units=256,
            dropout=0.25,
            from_logits=True
        ),
        large=dict(
            dense_units=512,
            dropout=0.5,
            from_logits=True
        ),
        xlarge=dict()
    )

    model_paces(
        model_fn,
        input_shape=input_shape,
        num_classes=num_classes,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        num_size_runs=num_size_runs,
        size_curves_epochs=size_curves_epochs,
        model_cfg=model_cfg,
        init_model_key=init_model_key,
        outpath=outpath
    )
