import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import random


# tf.random.set_seed(1234)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def fine_tune_model(base_model, dataset, lr=0.0001, epochs=10, seed=None, k=5):
    #
    # Load Datasets
    #
    if seed:
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
    
    #
    # Setup Model
    #
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy, 
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)
    ]

    base_model = base_cnn.build(use_resnet=True)
    base_model = base_cnn.compile(base_model, lr, metrics)

    img_b, lab_b = next(iter(train_ds))
    base_model.train_on_batch(img_b, lab_b)
    base_model.summary()

    base_model.save(os.path.join(root_model_dir, 'untrained'))

    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(root_model_dir, 'course'),
        monitor='val_sparse_categorical_accuracy',
        verbose=1,
        save_best_only=True,
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        patience=4,
        mode="max",
        min_delta=0.0025
    )
    log = tf.keras.callbacks.CSVLogger(
        filename=os.path.join(root_model_dir,'training.log'),
        append=True,
    )
    history = base_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[mc,es,log],
    )

    #
    # Fine Tuning
    #
    fine_tune_at = 100
    base_model.trainable=True
    for layer in base_model.layers[0].layers[:fine_tune_at]:
        layer.trainable = False 

    base_model = base_cnn.compile(base_model, lr*0.1)

    course_epochs = len(history.epoch)
    ft_epochs = 40

    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(root_model_dir, 'fine_tune'),
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        mode="max",
        min_delta=0.0025
    )
    log = tf.keras.callbacks.CSVLogger(
        filename=os.path.join(root_model_dir,'training.log'),
        append=True,
    )

    history_ft = base_model.fit(
        train_ds,
        epochs=course_epochs+ft_epochs,
        initial_epoch=course_epochs,
        validation_data=val_ds,
        callbacks=[mc, es, log],
    )

    test_loss, test_acc = base_model.evaluate(test_ds)
    test_file_path = os.path.join(root_model_dir,'fine_tune','test_results.log')
    with open(test_file_path, 'w+') as f:
        f.write('test_loss,test_accuracy\n{},{}'.format(test_loss, test_acc))

    return history, history_ft
    
