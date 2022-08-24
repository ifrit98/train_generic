import os
import sys
import time
import pickle
import logging
from xml.dom import NotFoundErr
import tensorflow as tf

from .eval import evaluate_model
from .stream_logger import StreamToLogger
from .utils import plot_metrics



def train(model_fn, model_cfg, data_fn, data_cfg, 
          plot_model=True, redirect_stdout=True,
          epochs=25, steps_per_epoch=None, saved_model_path='saved_model',
          checkpoint_path='model_ckpt/cp.ckpt', monitor='val_loss',
          hist_path='history/model_history', labels=None,
          stopping_patience=5, histogram_freq=5, profile_batch=0,
          lr_factor=0.5, verbose=True, lr_patience=3, min_learning_rate=1e-5):

    if data_fn is None: raise NotFoundErr(
        "`data_fn` must be a callable returning at least 2 tensorflow dataset objects.")
    if model_fn is None: raise NotFoundErr("Must provide `model_fn` callable.")
    if model_cfg == {}: raise NotFoundErr("Must provide `model_cfg` dict.")

    if redirect_stdout:
        logger = logging.getLogger('train')
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler('train_stdout.log')
        fh.setLevel(logging.DEBUG)

        fmt = '%(name)s - %(message)s'
        formatter = logging.Formatter(fmt)
        fh.setFormatter(formatter)

        logger.addHandler(fh)
        sys.stdout = StreamToLogger(logger, logging.DEBUG)

    start = time.time()

    # Instantiate model
    model = model_fn(**model_cfg)

    if plot_model:
        tf.keras.utils.plot_model(model, show_shapes=True)
        print("Model block diagram saved to {}/model.png".format(os.getcwd().upper()))
    print("Model created")

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    checkpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=True,
        monitor=monitor,
        verbose=verbose
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=lr_factor,
        patience=lr_patience, 
        min_lr=min_learning_rate
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=stopping_patience
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir='tensorboard',
        histogram_freq=histogram_freq,
        profile_batch=profile_batch
    )

    callbacks = [reduce_lr,  checkpt_cb, early_stop, tensorboard_cb]

    print("Callbacks loaded:")
    for cb in callbacks:
        print(cb)

    # Load data with data_fn and data_cfg
    datasets = data_fn(**data_cfg.get('data_loader_args', {}))
    if len(datasets) == 3:
        train_ds, val_ds, test_ds = datasets
    elif len(datasets) == 2:
        train_ds, test_ds = datasets
        val_ds = None
    else:
        raise ValueError("Must have at least 1 test set.")

    # TRAINING
    print("Begining training")
    history = model.fit(
            x=train_ds[0], y=train_ds[1],
            validation_data=tuple(val_ds),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1 if verbose else 0,
            steps_per_epoch=steps_per_epoch
        ) if isinstance(train_ds, list) else model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1 if verbose else 0,
            steps_per_epoch=steps_per_epoch
        )

    print("Training complete!")
    print("History:\n", history.history)

    # Save model, plots, and history objects
    model.save(saved_model_path)
    print("Model saved")
    print('\nTraining took {} seconds:'.format(int(time.time() - start)))

    hist_dir = os.path.dirname(hist_path)
    if not os.path.exists(hist_dir):
        os.mkdir(hist_dir)
    with open(hist_path, "wb") as f:
        pickle.dump(history.history, f)

    print("Plotting training curves...")
    plot_metrics(history)

    print("Evaluating model...")
    metadata = evaluate_model(model, test_ds, labels)
    return history, metadata


# Convenience function
def fit(model, ds, val_ds=None, epochs=10, callbacks=[], steps_per_epoch=None):
    return model.fit(
        ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch
    )


