import os
import yaml


from .lr_range_test import learn_rate_range_test
from .curve_tools import train_set_size_curves_tf, complexity_curves_tf
from .models.antirectifier import antirectifier_tiny
from .data.data_loader_mnist import mnist


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


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
        tiny=dict(
            dense_units=64,
            dropout=0.0,
            from_logits=True,
        ),
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
        xlarge=dict(
            dense_units=1024,
            dropout=0.5,
            from_logits=True
        )
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
