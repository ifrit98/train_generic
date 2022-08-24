import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Input, Flatten, Bidirectional, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import preprocessing


def bidirectional_lstm(input_shape=(None,), num_classes=1, max_features=64, activation='sigmoid'):
    # Input for variable-length sequences of integers
    inputs = Input(shape=input_shape, dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)
    # Add a classifier
    outputs = Dense(num_classes, activation=activation)(x)
    model = Model(inputs, outputs)
    model.summary()
    return model


def base_lstm_model(input_shape, layer_sizes, 
                    activation='tanh', 
                    recurrent_activation='hard_sigmoid',
                    dense_units=10, dense_activation='sigmoid',
                    l2_lambda=3e-2, learning_rate=5e-2, 
                    dropout=0.0, recurrent_dropout=0.0,
                    stateful=False, return_sequences=True,
                    return_state=False, unroll=False,
                    flatten_lstm_features=False, model_nm='base_lstm'):
    n_layers = len(layer_sizes)

    model = Sequential(name=model_nm)
    model.add(Input(input_shape))
    for i in range(n_layers-1):
        model.add(LSTM(
            units=layer_sizes[i],
            activation=activation, recurrent_activation=recurrent_activation,
            kernel_regularizer=l2(l2_lambda), recurrent_regularizer=l2(l2_lambda),
            dropout=dropout, recurrent_dropout=recurrent_dropout,
            return_sequences=return_sequences, return_state=return_state,
            stateful=stateful, unroll=unroll
        ))
        model.add(BatchNormalization())

    model.add(LSTM(
        units=layer_sizes[-1],
        activation=activation, recurrent_activation=recurrent_activation,
        kernel_regularizer=l2(l2_lambda), recurrent_regularizer=l2(l2_lambda),
        dropout=dropout, recurrent_dropout=recurrent_dropout,
        return_sequences=True if flatten_lstm_features else False, 
        return_state=False,
        stateful=False, unroll=False
    ))
    model.add(BatchNormalization())

    if flatten_lstm_features:
        model.add(Flatten())
        model.add(Dense((input_shape[0] * layer_sizes[-1]) // 8))

    model.add(Dense(units=dense_units, activation=dense_activation))

    model.compile(loss='binary_crossentropy',
                metrics=['accuracy'],
                optimizer=Adam(lr=learning_rate))

    print(model.summary())

    return model

model_configs = dict(
    base_lstm_small=dict(
        layer_sizes=[128,64,32],
        activation='tanh',
        recurrent_activation='hard_sigmoid',
        dense_activation='sigmoid',
        l2_lambda=3e-2,
        learning_rate=5e-2,
        dropout=0.0,
        recurrent_dropout=0.0,
        stateful=False,
        return_sequences=True,
        return_state=False,
        unroll=False,
        flatten_lstm_features=False,
    ),
    base_lstm_large=dict(
        layer_sizes=[256, 128, 64, 32, 16],
        activation='tanh',
        recurrent_activation='hard_sigmoid',
        dense_activation='sigmoid',
        l2_lambda=3e-2,
        learning_rate=5e-2,
        dropout=0.0,
        recurrent_dropout=0.0,
        stateful=False,
        return_sequences=True,
        return_state=False,
        unroll=False,
        flatten_lstm_features=True,
    ),
)


def build_base_lstm(input_shape, num_classes, model_nm='base_lstm_small'):
  cfg = model_configs[model_nm]

  model = base_lstm_model(
    input_shape=input_shape,
    layer_sizes=cfg['layer_sizes'],
    dense_units=num_classes,
    dense_activation=cfg['dense_activation'],
    activation=cfg['activation'],
    recurrent_activation=cfg['recurrent_activation'],
    return_state=cfg['return_state'],
    return_sequences=cfg['return_sequences'],
    stateful=cfg['stateful'],
    unroll=cfg['unroll'],
    flatten_lstm_features=cfg['flatten_lstm_features'],
    model_nm=model_nm
  )

  return model




# activation='tanh'
# recurrent_activation='hard_sigmoid'
# dense_units=10 
# dense_activation='sigmoid'
# l2_lambda=3e-2
# learning_rate=5e-2 
# dropout=0.0
# recurrent_dropout=0.0
# stateful=False
# return_sequences=True
# return_state=False
# unroll=False
# flatten_lstm_features=False
# layer_sizes=[128,64,32]
# input_shape=[100, 3143]

# # Define a learning rate decay method:
# lr_decay = ReduceLROnPlateau(monitor='loss', 
#                              patience=1, verbose=0, 
#                              factor=0.5, min_lr=1e-8)
# # Define Early Stopping:
# early_stop = EarlyStopping(monitor='val_acc', min_delta=0, 
#                            patience=30, verbose=1, mode='auto',
#                            baseline=0, restore_best_weights=True)

