import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


# REFERENCES #
# https://arxiv.org/abs/1712.09005
# https://distill.pub/2016/misread-tsne/
# https://stats.stackexchange.com/questions/263539/clustering-on-the-output-of-t-sne

product = lambda x: reduce(lambda a,b: a*b, x)


def correlation_heatmap(df, 
                        cols=None, mask_upper=True, show=True, 
                        light_cmap=False, lw=0.5, outpath=None):
    # Correlation between different variables
    df = df[cols] if cols is not None else df
    corr = df.corr()

    # Set up the matplotlib plot configuration
    fig, ax = plt.subplots(figsize=(12, 10))

    # Generate a mask for upper traingle
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None

    # Configure a custom diverging colormap
    cmap = "YlGnBu" if light_cmap else sns.diverging_palette(230, 20, as_cmap=True) 

    # Draw the heatmap
    sns.heatmap(corr, annot=True, mask = mask, cmap=cmap, linewidths=lw)

    if outpath is not None:
        plt.savefig(outpath)

    # Show and return fig,ax
    if show:
        plt.show()

    return fig, ax 


def pca(data, labels=None, n_components=2, whiten=False, 
        random_state=None, show=True, outpath=None, title=None):

    # Create PCA transform
    p = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    pca_transform = p.fit_transform(data)
    
    # Show cumulative explained variation (cumsum of component variances)
    print('Cumulative explained variation for {} principal components: {}'.format(
        n_components, np.sum(p.explained_variance_ratio_))
    )
    pcaratio = p.explained_variance_ratio_

    if show:
        ax = sns.scatterplot(x=np.arange(len(pcaratio)), y=np.cumsum(pcaratio))
        ax.set_title("PCA Explained Variance")
        plt.legend()
        plt.show()

        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x=pca_transform[:,0], y=pca_transform[:,1],
            hue=None if labels is None else labels,
            palette=sns.color_palette("hls", len(np.unique(labels))),
            legend="full",
            alpha=0.3
        )
        plt.title(title or "PCA {} components".format(n_components))
        if outpath is not None:
            plt.savefig(outpath)        
        plt.show()
    
    return pca_transform


def tsne(x_train, y_train=None, 
         random_state=123, 
         scale=False,
         scale_type='standard',
         perplexity=50,
         n_components=2, 
         verbose=1, 
         n_iter=2000,
         early_exaggeration=12, 
         n_iter_without_progress=1000,
         show=True,
         outpath=None,
         title="T-SNE projection"):

    # Ensure data is in form (n_obs, samples)
    # e.g. if x_train.shape == (n_obs, samples_x, samples_y), 
    # Will be reshaped -> (n_obs, samples_x*sampels_y)
    if len(x_train.shape) > 2:
        x_train = np.reshape(
            x_train, [x_train.shape[0]] + [product(x_train.shape[1:])]
        )

    # Use a scaler to standardize data
    if scale:
        scaler_fn = {
            'normal': Normalizer(),
            'minmax': MinMaxScaler(),
            'standard': StandardScaler()
        }[scale_type]
        x_train = scaler_fn.fit_transform(x_train)

    # Create and Run T-SNE with given hparams
    tsne = TSNE(
        n_components=n_components, verbose=verbose, random_state=random_state,
        perplexity=perplexity, early_exaggeration=early_exaggeration, n_iter=n_iter,
        n_iter_without_progress=n_iter_without_progress
    )
    tsne_transform = tsne.fit_transform(x_train)


    if show:
        # Use pandas for plotting convenience
        df = pd.DataFrame()
        df["tsne-1"] = tsne_transform[:,0]
        df["tsne-2"] = tsne_transform[:,1]
        df["y"]      = y_train

        # Create scatterplot and show
        if y_train is None:
            ax = sns.scatterplot(
                x="tsne-1", y="tsne-2",
                palette=sns.color_palette("hls", 10), 
                data=df
            )
        else:
            ax = sns.scatterplot(
                x="tsne-1", y="tsne-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", len(np.unique(y_train))), 
                data=df
            )
        ax.set_title(title)
        if outpath is not None:
            plt.savefig(outpath)
        plt.show()

    del df
    return tsne_transform


def pca_then_tsne(x_train, y_train=None, 
                  n_pca_components=50, 
                  n_tsne_components=2, 
                  whiten=False, 
                  perplexity=50, # 40
                  verbose=1,
                  early_exag=12,
                  n_iter=2000, # 300
                  n_iter_without_progress=1000,
                  show=True,
                  outpath=None,
                  title="PCA->TSNE"):

    # Compute PCA and get transform
    pca_transform = pca(x_train, n_components=n_pca_components, whiten=whiten, show=False)

    # Compute TSNE and get transform
    tsne_transform = tsne(
        x_train=pca_transform, y_train=y_train,
        n_components=n_tsne_components, verbose=verbose, perplexity=perplexity, 
        n_iter=n_iter, early_exaggeration=early_exag, 
        n_iter_without_progress=n_iter_without_progress, show=show
    )

    # Final visualization
    if show:
        num_classes = len(np.unique(y_train))
        df = pd.DataFrame()
        df['pca-one'] = pca_transform[:,0]
        df['pca-two'] = pca_transform[:,1]
        df['tsne-pca50-one'] = tsne_transform[:,0]
        df['tsne-pca50-two'] = tsne_transform[:,1]
        df['y'] = y_train

        plt.figure(figsize=(16,8))
        ax1 = plt.subplot(1, 2, 1)
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue=None if y_train is None else "y",
            palette=sns.color_palette("hls", num_classes),
            data=df,
            legend="full",
            alpha=0.3,
            ax=ax1
        )
        ax1.set_title("PCA {} components".format(n_pca_components))
        ax2 = plt.subplot(1, 2, 2)
        sns.scatterplot(
            x="tsne-pca50-one", y="tsne-pca50-two",
            hue=None if y_train is None else "y",
            palette=sns.color_palette("hls", num_classes),
            data=df,
            legend="full",
            alpha=0.3,
            ax=ax2
        )
        ax2.set_title("TSNE {} components".format(n_tsne_components))
        plt.title(title or "PCA ({}) -> TSNE ({})".format(n_pca_components, n_tsne_components))
        plt.tight_layout()
        if outpath is not None:
            plt.savefig(outpath)
        plt.show()

    return tsne_transform



def pca_3D(X, y, n_components=3, show=True, outpath=None):
    """
    Compute PCA with 3 components and visualize in 3Space
    """
    #
    # Compute PCA transform for 3D axes
    #
    transform = pca(X, n_components=n_components, show=False)

    if show:
        # Create dataframe for plotular convenience 
        cols = list(map(lambda x: 'pca-' + str(x), range(1, transform.shape[-1] + 1)))
        df = pd.DataFrame(transform, columns=cols)
        df['y'] = y
        
        # Do the plotting
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(
            xs=df["pca-1"],
            ys=df["pca-2"],
            zs=df["pca-3"],
            c=None if y is None else df["y"],
            cmap='tab10'
        )
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')
        if outpath is not None:
            plt.savefig(outpath)
        plt.show()

    return transform


def dbscan():
    raise NotImplementedError


def demo():
    from mlcurves.cluster import pca_3D, pca, tsne, pca_then_tsne
    from mlcurves.curve_utils import mnist_npy

    X, y = mnist_npy(return_test=False, shuffle=False)


    _ = pca(X, y)


    _ = pca_3D(X, y)


    _ = tsne(X, y)


    _ = pca_then_tsne(X, y)
