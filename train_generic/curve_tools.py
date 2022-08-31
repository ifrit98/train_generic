import os
import numpy as np
from inspect import signature
fargs = lambda f: list(signature(f).parameters.keys())

from .plot_utils import plot_metrics


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

