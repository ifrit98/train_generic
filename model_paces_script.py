import os

from mlcurves import learn_rate_range_test
from mlcurves import complexity_curves_tf
from mlcurves import train_set_size_curves_tf
from sympy import dsolve

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.metrics import SparseCategoricalAccuracy, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy




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
                 outpath='training_curves_' + timestamp()):
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
                             trainset, valset, 
                             testset, num_classes,
                             batch_size=16,
                             epochs=10, n_runs=11, 
                             shuffle_init=True, 
                             buffer_size=None,
                             outpath='./plot'):
    # Takes model and tensorflow Models and data.Dataset objects ONLY
    # Assume model is compiled properly before being passed as an argument

    # Try unbatching first
    try:
        trainset = trainset.unbatch()
        valset   = valset.unbatch()
        testset  = testset.unbatch()
    except:
        pass

    # Get monotonically increasing range based on percentages
    train_len = len(list(trainset.as_numpy_iterator()))

    if not os.path.exists(outpath): os.mkdir(outpath)

    if buffer_size is None:
        buffer_size = train_len

    if shuffle_init:
        trainset = trainset.shuffle(buffer_size)
        valset = valset.shuffle(buffer_size)
        testset = testset.shuffle(buffer_size)

    u = 1 / n_runs
    rng = np.arange(u, 1+u, u)

    # Split datasets into random subsets of increasing size, e.g. [10%, 20%,..., 100%]
    train_sizes = [int(train_len * p) for p in rng]
    tr_prev = 0

    if model_args.get('input_shape') is None or model_args.get('num_classes') is None:
        for x in trainset: break
        model_args['input_shape'] = x[0].shape
        model_args['num_classes'] = num_classes
        del x

    histories = {}
    results = {}
    for i, tr in enumerate(train_sizes):
        print("Starting dataset size: (train) {}", tr)
        print("Percentage of full trainset {}%".format((tr/train_len)*100))

        ds_sub = trainset.skip(tr_prev).take(tr).shuffle(buffer_size).batch(batch_size)
        tr_prev = tr

        model = model_fn(**model_args)
        history = model.fit(ds_sub, validation_data=valset.batch(batch_size), epochs=epochs)
        histories[i] = history

        res = model.evaluate(testset.batch(batch_size))
        results[i] = res
        
        plot_metrics(history, show=False, outpath=os.path.join(
            outpath, 'training_curves_{}'.format(i))
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
        outpath=os.path.join(outpath, 'final_train_size_curves.png')
    )
    return total_history




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
            model = model_fn(
                input_shape=input_shape, num_classes=num_classes, **cfg
            )

        h = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=epochs,
            batch_size=batch_size
        )
        histories[i] = h
        plot_metrics(h, show=False, outpath=os.path.join(
            outpath, 'training_curves_{}'.format(i))
        )

        r = model.evaluate(test_ds)
        results[i] = r

        model_sizes[i] = get_param_count(model)

    model_basename = nm.split("_")[0]
    total_history = process_history(results, histories)

    plot_metrics(
        total_history, show=False, xlab="Model complexity (# params)", 
        ylab="Crossentropy Loss",
        xticks=range(len(model_sizes)), xtick_labels=list(model_sizes.values()),
        outpath=os.path.join(outpath, '{}_complexity_curves.png'.format(model_basename))
    )

    return total_history



import yaml
def model_paces(model_fn, 
                input_shape, 
                num_classes, 
                train_ds, 
                val_ds,
                test_ds, 
                num_size_runs=11,
                init_model_key='baseline',
                model_cfg='model_cfg.yaml', # can be a python dict()
                outpath="."):
    """
    Put a model through its paces.

    (1) Runs a learning rate scheduler to find best learning rate parameters for this model, 
    given a particular dataset.

    (2) Runs a routine to train and test model on increasing data set sizes (takes subsets)

    (3) Runs a rounine to infer average best model size (measured in # parameters)

    (4) Returns the results as `dict` and saves plots to `outpath/*.png`
    """

    if isinstance(model_cfg, str):
        with open(model_cfg, 'rb') as f:
            model_cfg = yaml.load(f)
    
    model_args = model_cfg.get(init_model_key, {})
    model_args.update(dict(input_shape=input_shape, num_classes=num_classes))

    model = model_fn(**model_args) # model_fn(input_shape, num_classes)
    print(model.summary())

    (min_lr, init_lr), h = learn_rate_range_test(
        model, train_ds, outpath=os.path.join(outpath, "lr_range_test")
    )

    train_size_history = train_set_size_curves_tf(
        model_fn, model_args, train_ds, val_ds, test_ds, 
        num_classes=num_classes, epochs=25, n_runs=num_size_runs,
        outpath=os.path.join(outpath, "train_size_test")
    )

    complexity_history = complexity_curves_tf(
        model_fn, configs=model_cfg, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
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



def paces_demo(outpath='./out', n_val=2000):
    import os
    from mlcurves.curve_utils import mnist

    ds, ts = mnist(shuffle=True, vectorize=True, expand_dims=False, batch_size=16)
    vs = ts.take(n_val)
    ts = ts.skip(n_val)

    for x in ds: break
    input_shape = x[0].shape[1:]
    num_classes = 10
    print("input shape:", input_shape)

    from mlcurves.models.antirectifier import build_antirectifier_dense, dense_configs
    from mlcurves import model_paces

    paces = model_paces(
        build_antirectifier_dense, 
        input_shape,
        num_classes=num_classes,
        train_ds=ds, test_ds=ts, val_ds=vs,
        model_cfg=dense_configs,
        outpath=os.path.join(outpath, "model_paces")
    )
    print("model paces results: {}".format(paces))



from train_generic import mnist

train_ds, val_ds, test_ds = mnist(
    expand_last_dim=False, subsample=True, batch_size=16, 
    one_hot_labels=False,
    return_val_set=True,
    drop_remainder=True
)
trainset=train_ds
valset=val_ds
testset=test_ds

model_fn=antirectifier_tiny
input_shape=(784,)
num_classes=10
init_model_key='small'
epochs=10
n_runs=3
shuffle_init=True
outpath='./'

model_cfg = dict(
    small=dict(
        dense_units=128,
        dropout=0.1,
        from_logits=True
    ),
    baseline=dict(
        dense_units=256,
        dropout=0.25,
        from_logits=False
    ),
    large=dict(
        dense_units=512,
        dropout=0.5,
        from_logits=False
    ),
)
