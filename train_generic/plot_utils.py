import matplotlib.pyplot as plt
import seaborn as sns
from warnings import warn
import time


timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))

# Plotting routine for history objects
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
        acc  = history.history['acc']
    elif 'accuracy' in keys:
        acc  = history.history['accuracy']
    elif 'train_acc' in keys:
        acc  = history.history['train_acc']
    elif 'train_accuracy' in keys:
        acc  = history.history['train_accuracy']
    else:
        raise ValueError("Training accuracy not found")


    plt.plot(epochs, acc, color='green', 
        marker='+', linestyle='dashed', label='Training accuracy'
    )

    if 'val_acc' in keys:
        val_acc = history.history['val_acc']
    elif 'val_accuracy' in keys:
        val_acc = history.history['val_accuracy']
    else:
        raise ValueError("Validation accuracy not found")

    plt.plot(epochs, val_acc, color='blue', 
        marker='o', linestyle='dashed', label='Validation accuracy'
    )

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
    elif 'validation_loss' in keys:
        val_loss = history.history['validation_loss']
    else:
        raise ValueError("Validation loss not found")

    plt.plot(epochs, val_loss, color='blue', 
        marker='o', linestyle='dashed', label='Validation Loss'
    )

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
