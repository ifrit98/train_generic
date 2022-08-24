# Bayesian classifier
# https://github.com/kyle-dorman/bayesian-neural-network-blogpost

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
K = tf.keras.backend


# standard categorical cross entropy
# N data points, C classes
# true - true values. Shape: (N, C)
# pred - predicted values. Shape: (N, C)
# returns - loss (N)
def categorical_cross_entropy(true, pred):
    return tf.reduce_sum(true * K.log(pred), axis=1)


# Bayesian categorical cross entropy.
# N data points, C classes, T monte carlo simulations
# true - true values. Shape: (N, C)
# pred_var - predicted logit values and variance. Shape: (N, C + 1)
# returns - loss (N,)
def bayesian_categorical_crossentropy(T, num_classes):

    iterable = K.variable(K.ones(T))

    def bayesian_categorical_crossentropy_internal(true, pred_var):
        true = tf.cast(true, 'float32')
        # shape: (N,)
        std = K.sqrt(pred_var[:, num_classes])
        # shape: (N,)
        variance = pred_var[:, num_classes]
        variance_depressor = K.exp(variance) - K.ones_like(variance)
        # shape: (N, C)
        pred = pred_var[:, 0:num_classes]
        # shape: (N,)
        undistorted_loss = K.categorical_crossentropy(true, pred, from_logits=True)
        # shape: (T,)
        dist = tfp.distributions.Normal(loc=K.zeros_like(std), scale=std)
        monte_carlo_results = K.map_fn( 
            gaussian_categorical_crossentropy(
                true, pred, dist, undistorted_loss, num_classes),
            iterable, 
            name='monte_carlo_results')
        
        variance_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss
        
        return variance_loss + undistorted_loss + variance_depressor
    return bayesian_categorical_crossentropy_internal

# for a single monte carlo simulation, 
#   calculate categorical_crossentropy of 
#   predicted logit values plus gaussian 
#   noise vs true values.
# true - true values. Shape: (N, C)
# pred - predicted logit values. Shape: (N, C)
# dist - normal distribution to sample from. Shape: (N, C)
# undistorted_loss - the crossentropy loss without variance distortion. Shape: (N,)
# num_classes - the number of classes. C
# returns - total differences for all classes (N,)
def gaussian_categorical_crossentropy(true, 
                                     pred, 
                                     dist, 
                                     undistorted_loss, 
                                     num_classes):
    def map_fn(i):
        std_samples = K.transpose(dist.sample(num_classes))
        distorted_loss = K.categorical_crossentropy(true, pred + std_samples, from_logits=True)
        diff = undistorted_loss - distorted_loss
        return -K.elu(diff)
    return map_fn

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

#  TODO: This dude fucked up and probably never tested this.  Rework this.
def resnet50(input_tensor):
	base_model = tf.keras.applications.ResNet50(include_top=False, input_tensor=input_tensor)
	# freeze encoder layers to prevent over fitting
	for layer in base_model.layers:
		layer.trainable = False

	output_tensor = Flatten()(base_model.output)
	return output_tensor #Model(inputs=input_tensor, outputs=output_tensor)

def create_bayesian_model(input_shape, output_classes):
    encoder_input_tensor = Input(shape=input_shape)
    encoder_out = resnet50(encoder_input_tensor)
	# input_tensor = Input(shape=encoder_out.shape[1:])
    x = BatchNormalization(name='post_encoder')(encoder_out)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    logits = Dense(output_classes)(x)
    variance_pre = Dense(1)(x)
    variance = Activation('softplus', name='variance')(variance_pre)
    logits_variance = concatenate([logits, variance], name='logits_variance')
    softmax_output = Activation('softmax', name='softmax_output')(logits)

    model = Model(inputs=encoder_input_tensor, outputs=[logits_variance,softmax_output])
    return model


if False:
    model = create_bayesian_model([28, 28, 3], 10)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3, decay=0.001),
        loss={
        'logits_variance': bayesian_categorical_crossentropy(100, 10),
        'softmax_output': 'categorical_crossentropy'
        },
        metrics={'softmax_output': tf.keras.metrics.categorical_accuracy},
        loss_weights={'logits_variance': .2, 'softmax_output': 1.})

    model.summary()

    keras = tf.keras
    # Training parameters
    batch_size = 128
    num_classes = 10
    epochs = 5

    # The data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # x_train = x_train.reshape(-1, 784)
    # x_test = x_test.reshape(-1, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    x_train = tf.tile(tf.expand_dims(x_train, -1), [1, 1, 1, 3])
    x_test = tf.tile(tf.expand_dims(x_test, -1), [1, 1, 1, 3])


    # Train the model
    model.fit(
        x_train, tf.one_hot(y_train, num_classes, dtype='int32'), 
        batch_size=batch_size, epochs=epochs, validation_split=0.15)

    # Test the model
    model.evaluate(x_test, tf.one_hot(y_test, num_classes))
