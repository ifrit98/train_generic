import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns



timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))
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
    idx = np.argmin(history.history['loss'])
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
    import tensorflow as tf
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


