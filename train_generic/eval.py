import time
import pickle
from math import sqrt
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt


# TODO: import this from style file in frontrow project toplevel
# that way we don't violate DRY
#Defaults for legible figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams["image.cmap"] = 'jet'

from sklearn.metrics import roc_curve, RocCurveDisplay, auc
from sklearn.metrics import confusion_matrix, classification_report

from itertools import cycle
from scipy import interp

from .cluster import pca, tsne, pca_then_tsne
from .utils import is_tensor

# Hex values
PHASE1_colors = dict([
    ('cargo', '#eda042'), 
    ('negative', '#eb52ff'), 
    ('passenger', '#758cc0'), 
    ('towing', '#1e6d2b'), 
    ('tug', '#3ee1d1')
])

def divisors(n):
    large_divisors = []
    for i in range(1, int(sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor

def load_eval_results(path='./eval_metrics.pickle'):
    with open(path, 'rb') as f:
        ev = pickle.load(f)
    return ev

def sklearn_confusion_matrix_multi(y_pred, y_true):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    num_classes = y_pred.shape[-1] if len(y_pred.shape) > 1 else 1
    f, axes = plt.subplots(num_classes)
    axes = axes.ravel()
    for i in range(num_classes):
        disp = ConfusionMatrixDisplay(confusion_matrix(y_true[:, i],
                                                    y_pred[:, i]),
                                    display_labels=[0, i])
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(f'class {i}')
        disp.ax_.set_xlabel('')
        disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.show()

def convert_multilabel_to_categorical(y_pred):
    pred = np.asarray(list(map(lambda x: np.argmax(x), y_pred)))
    if is_tensor(y_pred):
        y_pred = y_pred.numpy()
    def map_fn(argmax):
        target = np.zeros(y_pred.shape[1], dtype='int32')
        target[argmax] = 1
        return target
    y_pred = np.asarray(list(map(map_fn, pred)))
    return y_pred


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    import pandas as pd
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Note that due to returning the created figure object, when this funciton is called in a
    notebook the figure willl be printed twice. To prevent this, either append ; to your
    function call, or modify the function by commenting out the return expression.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def generate_roc_curve(y_test, y_pred, labels, model_str):

    n_classes = len(labels)
    y_test = y_test.numpy() if is_tensor(y_test) else y_test
    y_score = y_pred.numpy() if is_tensor(y_pred) else y_pred

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro ROC (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro ROC (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, label, color in zip(range(n_classes), labels, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='{0} ({1:0.2f})'
                ''.format(label, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC: {} model'.format(model_str))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('ROC_{}_categorical.png'.format(model_str))
    plt.clf()
    plt.cla()

    return fpr, tpr, roc_auc


def gen_roc_phase1_like(y_test, 
                        y_pred, 
                        labels=None, 
                        model_str="model", 
                        colors=['#eda042', '#eb52ff', '#758cc0', '#1e6d2b', '#3ee1d1']):

    n_classes = len(labels)
    y_test = y_test.numpy() if is_tensor(y_test) else y_test
    y_score = y_pred.numpy() if is_tensor(y_pred) else y_pred

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()

    colors = cycle(colors) # cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, label, color in zip(range(n_classes), labels, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='{0} ({1:0.2f})'
                ''.format(label, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC: {} model'.format(model_str))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('ROC_{}_categorical_PH1.png'.format(model_str))
    plt.clf()
    plt.cla()

    return fpr, tpr, roc_auc


from warnings import warn
def evaluate_model(model, test_ds, labels, num_classes, model_str='model'):


    if isinstance(test_ds, list):
        batch_size = int(list(divisors(test_ds[0].shape[0]))[1])
        if batch_size > test_ds[0].shape[0] // 4:
            batch_size = 1
        test_ds[0] = np.reshape(
            test_ds[0], [-1, batch_size] + list(test_ds[0].shape[1:]))
        test_ds[1] = np.reshape(
            test_ds[1], [-1, batch_size] + list(test_ds[1].shape[1:]))
        test_ds = zip(test_ds[0], test_ds[1])

    # Made predictions with model
    variance, trues, preds = [], [], []
    x_test = []
    start = time.time()
    for i, x in enumerate(test_ds):
        x_test.append(x[0])
        if i % 10 == 0:
            print("Predicting test batch {}".format(i))
        y_pred = model.predict(x[0])
        y_true = x[1]
        preds.append(y_pred + 1e-8) # epsilon for numerical stability
        trues.append(y_true)
    end = time.time()
    print("Done making Predictions!")
    print("Took {} sec...".format(end-start))
    y_pred = tf.squeeze(tf.concat(preds, 0))
    y_true = tf.concat(trues, 0)    
    print("original y_true", y_true.shape)
    print("original y_pred", y_pred.shape)

    is_onehot = lambda y: y.shape[-1] == num_classes
    def return_both_onehot_and_int(x):
        if is_onehot(x):
            return x, tf.argmax(x, axis=-1)
        return tf.one_hot(x, depth=num_classes, dtype=x.dtype), x

    y_true_onehot, y_true_int = return_both_onehot_and_int(y_true)
    print("y_true_onehot:", y_true_onehot.shape)
    print("y_true_int", y_true_int.shape)

    # Cluster predictions
    x_test = np.squeeze(np.concatenate(x_test, 0))
    if len(x_test.shape) > 2:
        x_test = np.reshape(x_test, [-1, x_test.shape[-1]])

    y_true_cluster = y_true_int if y_true_int.shape[0] == x_test.shape[0] else np.reshape(
        np.squeeze(y_true_int), [-1]
    )
    y_pred_cluster = np.reshape(
        np.argmax(y_pred, axis=-1), [-1]
    ) if y_pred.shape[-1] == num_classes else y_pred
    print("after x_test", x_test.shape)
    print("after y_pred_cluster", y_pred_cluster.shape)
    print('after y_true_cluster', y_true_cluster.shape)

    try:
        plt.clf(); plt.cla();
        tsne_pred_transform = tsne(
            x_test, y_pred_cluster, 
            outpath='tsne_pred.png'
        ); plt.clf(); plt.cla()
        _ = tsne(x_test, y_true_cluster, outpath='tsne_true.png'); plt.clf(); plt.cla()
    except:
        tsne_pred_transform = None
        warn("TSNE clustering error...")

    try:
        pca_pred_transform = pca(
            x_test, y_pred_cluster, outpath='pca_pred.png'
        ); plt.clf(); plt.cla()
        _  = pca(x_test, y_true_cluster, outpath='pca_true.png'); plt.clf(); plt.cla()
    except:
        pca_pred_transform = None
        warn("PCA clustering error...")

    try:
        pca_then_tsne_pred_transform  = pca_then_tsne(
            x_test, y_pred_cluster, outpath='pca_then_tsne_pred.png', n_pca_components=10
        ); plt.clf(); plt.cla()
        _  = pca_then_tsne(
            x_test, y_true_cluster, n_pca_components=10,
            outpath='pca_then_tsne_true.png'); plt.clf(); plt.cla()
    except:
        pca_then_tsne_pred_transform = None
        warn("PCA->TSNE cluster error...")

    y_true_roc = np.reshape(y_true_onehot, [-1, num_classes])
    y_pred_roc = np.reshape(y_pred, [-1, num_classes])
    try:
        print("Plotting ROC curve")
        fpr, tpr, roc_auc = generate_roc_curve(
            y_true_roc, y_pred_roc, labels, model_str
        )
    except ValueError as e:
        fpr, tpr, roc_auc = [None] * 3
        print("failure in generating ROC curves... {}".format(e))

    # Plot ROC like phase 1 (clean version)
    try:
        print("Plotting phase1-like ROC curve...")
        gen_roc_phase1_like(y_true_roc, y_pred_roc, labels, model_str)
    except ValueError as e:
        print("...Failed {}".format(e))
        pass

    labels_dict = dict(zip(list(range(len(labels))), labels))
    labels = np.asarray(list(labels_dict.values())) if labels is None else labels
    print("LABELS:", labels)
    print("\nLABELS_DICT:", labels_dict)

    def decode_labels(arr, labels_dict):
        if len(np.asarray(arr).shape) == 1:
            return np.asarray([labels_dict[x] for x in arr])
        raise ValueError("Multiple labels... use different func to decode labels")

    if is_tensor(y_true):
        y_true = y_true.numpy()
    if is_tensor(y_pred):
        y_pred = y_pred.numpy()

    # Generate Classification report (F1 score) and Confusion Matrix
    true = decode_labels(np.argmax(y_true_roc, -1), labels_dict)
    pred = decode_labels(y_pred_cluster, labels_dict)
    mat = confusion_matrix(true, pred)
    report = classification_report(true, pred)
    print(report)
    print(mat)

    # Generate Heatmap
    plt.clf(); plt.cla();
    heatmap = sns.heatmap(mat/np.sum(mat), annot=True, fmt='.2%', cmap='Blues')
    list(map(lambda x: x[0].set_text(x[1]), zip(heatmap.yaxis.get_ticklabels(), labels)))
    list(map(lambda x: x[0].set_text(x[1]), zip(heatmap.xaxis.get_ticklabels(), labels)))
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=8)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("confusion_matrix_plot.png")
    print("\nFinished evaulating model.")

    # Save off evaluation results
    eval_metrics = {
        'false_pos_rate': fpr,
        'true_pos_rate': tpr,
        'variance': variance,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_true': y_true,
        'labels': labels,
        'labels_dict': labels_dict,
        'conf_mat': mat,
        'class_report': report,
        'tsne_pred_transform': tsne_pred_transform,
        'pca_pred_transform': pca_pred_transform,
        'pca_then_tsne_pred_transform': pca_then_tsne_pred_transform
    } 
    with open("eval_metrics.pickle", "wb") as f:
        pickle.dump(eval_metrics, f)
    with open("eval_metrics.md", "w") as f:
        f.writelines('\n'.join(list(eval_metrics)))

    return eval_metrics





def evaluate_model_OG(model, test_ds, labels, num_classes, model_str='model'):

    labels = labels or list(range(num_classes)) # if labels=None

    if isinstance(test_ds, list):
        batch_size = int(list(divisors(test_ds[0].shape[0]))[1])
        if batch_size > test_ds[0].shape[0] // 4:
            batch_size = 1
        test_ds[0] = np.reshape(
            test_ds[0], [-1, batch_size] + list(test_ds[0].shape[1:]))
        test_ds[1] = np.reshape(
            test_ds[1], [-1, batch_size] + list(test_ds[1].shape[1:]))
        test_ds = zip(test_ds[0], test_ds[1])

    # Made predictions with model
    variance, trues, preds = [], [], []
    x_test = []
    start = time.time()
    for i, x in enumerate(test_ds):
        x_test.append(x[0])
        if i % 10 == 0:
            print("Predicting test batch {}".format(i))
        y_pred = model.predict(x[0])
        y_true = x[1]
        preds.append(y_pred + 1e-8) # epsilon for numerical stability
        trues.append(y_true)
    end = time.time()
    print("Done making Predictions!")
    print("Took {} sec...".format(end-start))
    y_pred = tf.squeeze(tf.concat(preds, 0))
    y_true = tf.concat(trues, 0)


    # Cluster predictions
    x_test = np.squeeze(np.concatenate(x_test, 0))
    if len(x_test.shape) > 2:
        x_test = np.reshape(x_test, [-1, x_test.shape[-1]])

    if len(y_true.shape) > 2:
        y_true = np.reshape(y_true, [-1, y_true.shape[-1]])

    if len(y_pred.shape) > 2:
        y_pred = np.reshape(y_pred, [-1, y_pred.shape[-1]])


    y_true_cluster = y_true if len(y_true.shape) < 2 else np.argmax(np.squeeze(y_true), 1)

    try:
        plt.clf(); plt.cla();
        tsne_pred_transform = tsne(
            x_test, np.argmax(y_pred, 1), outpath='tsne_pred.png'
        ); plt.clf(); plt.cla()
        _ = tsne(x_test, y_true_cluster, outpath='tsne_true.png'); plt.clf(); plt.cla()
    except:
        tsne_pred_transform = None
        warn("TSNE clustering error...")

    try:
        pca_pred_transform = pca(
            x_test, np.argmax(y_pred, 1), outpath='pca_pred.png'
        ); plt.clf(); plt.cla()
        _  = pca(x_test, y_true_cluster, outpath='pca_true.png'); plt.clf(); plt.cla()
    except:
        pca_pred_transform = None
        warn("PCA clustering error...")

    try:
        pca_then_tsne_pred_transform  = pca_then_tsne(
            x_test, np.argmax(y_pred, 1), outpath='pca_then_tsne_pred.png'
        ); plt.clf(); plt.cla()
        _  = pca_then_tsne(x_test, y_true_cluster, outpath='pca_then_tsne_true.png'); plt.clf(); plt.cla()
    except:
        pca_then_tsne_pred_transform = None
        warn("PCA->TSNE cluster error...")

    # Generate ROC curves (TODO: Update with lab code that Jon fixed)
    y_true_roc = np.squeeze(tf.one_hot(
        y_true, num_classes, dtype='int32').numpy() if len(y_true.shape) == 1 else y_true
    )
    try:
        print("Plotting ROC curve")
        fpr, tpr, roc_auc = generate_roc_curve(
            y_true_roc, 
            y_pred, labels, model_str
        )
    except ValueError as e:
        fpr, tpr, roc_auc = [None] * 3
        print("failure in generating ROC curves... {}".format(e))

    # Plot ROC like phase 1 (clean version)
    try:
        print("Plotting phase1-like ROC curve...")
        gen_roc_phase1_like(y_true_roc, y_pred, labels, model_str)
    except ValueError as e:
        print("...Failed {}".format(e))
        pass


    labels_dict = dict(zip(list(range(len(labels))), labels))
    labels = np.asarray(list(labels_dict.values())) if labels is None else labels
    print("LABELS:", labels)
    print("\nLABELS_DICT:", labels_dict)

    def decode_labels(arr, labels_dict):
        if len(np.asarray(arr).shape) == 1:
            return np.asarray([labels_dict[x] for x in arr])
        raise ValueError("Multiple labels... use different func to decode labels")

    if is_tensor(y_true):
        y_true = y_true.numpy()
    if is_tensor(y_pred):
        y_pred = y_pred.numpy()

    # Generate Classification report (F1 score) and Confusion Matrix
    pred = np.asarray(list(map(lambda x: np.argmax(x), y_pred)))
    true = np.asarray(list(map(lambda x: np.argmax(x), y_true))) \
        if len(y_true.shape) > 1 else y_true 
    
    true = decode_labels(true, labels_dict)
    pred = decode_labels(pred, labels_dict)
    mat = confusion_matrix(true, pred)
    report = classification_report(true, pred)
    print(report)
    print(mat)

    # Generate Heatmap
    plt.clf(); plt.cla();
    heatmap = sns.heatmap(mat/np.sum(mat), annot=True, fmt='.2%', cmap='Blues')
    list(map(lambda x: x[0].set_text(x[1]), zip(heatmap.yaxis.get_ticklabels(), labels)))
    list(map(lambda x: x[0].set_text(x[1]), zip(heatmap.xaxis.get_ticklabels(), labels)))
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=8)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=8)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("confusion_matrix_plot.png")
    print("\nFinished evaulating model.")

    # Save off evaluation results
    eval_metrics = {
        'false_pos_rate': fpr,
        'true_pos_rate': tpr,
        'variance': variance,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_true': y_true,
        'labels': labels,
        'labels_dict': labels_dict,
        'conf_mat': mat,
        'class_report': report,
        'tsne_pred_transform': tsne_pred_transform,
        'pca_pred_transform': pca_pred_transform,
        'pca_then_tsne_pred_transform': pca_then_tsne_pred_transform
    } 
    with open("eval_metrics.pickle", "wb") as f:
        pickle.dump(eval_metrics, f)
    with open("eval_metrics.md", "w") as f:
        f.writelines('\n'.join(list(eval_metrics)))

    return eval_metrics

