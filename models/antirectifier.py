
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, MaxPooling1D, Dense, Flatten
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D



class Antirectifier(tf.keras.layers.Layer):
  """Build simple custome layer."""

  def __init__(self, initializer="he_normal", **kwargs):
    super(Antirectifier, self).__init__(**kwargs)
    self.initializer = tf.keras.initializers.get(initializer)

  def build(self, input_shape):
    output_dim = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(output_dim * 2, output_dim),
        initializer=self.initializer,
        name="kernel",
        trainable=True,
    )

  def call(self, inputs):  #pylint: disable=arguments-differ
    inputs -= tf.reduce_mean(inputs, axis=-1, keepdims=True)
    pos = tf.nn.relu(inputs)
    neg = tf.nn.relu(-inputs)
    concatenated = tf.concat([pos, neg], axis=-1)
    mixed = tf.matmul(concatenated, self.kernel)
    return mixed

  def get_config(self):
    # Implement get_config to enable serialization. This is optional.
    base_config = super(Antirectifier, self).get_config()
    config = {"initializer": tf.keras.initializers.serialize(self.initializer)}
    return dict(list(base_config.items()) + list(config.items()))


def antirectifier_tiny(input_shape, num_classes=10, dense_units=256, dropout=0.2):
  model = Sequential(
      [
          Input(shape=input_shape),
          Dense(dense_units),
          Antirectifier(),
          Dense(dense_units),
          Antirectifier(),
          Dropout(dropout),
          Dense(num_classes),
      ]
  )
  model.compile(
      loss=SparseCategoricalCrossentropy(from_logits=True),
      optimizer=RMSprop(),
      metrics=[SparseCategoricalAccuracy()],
  )
  return model


def antirectifier_dense(input_shape,
                        num_classes,
                        model_nm='antirect_base',
                        n_layers=4,
                        unit_sizes=[64, 128, 256, 512],
                        dropout_rate=0.2,
                        flatten=True,
                        logits=False,
                        jit_compile=False):

    assert len(unit_sizes) == n_layers

    model = Sequential(name=model_nm)
    model.add(Input(input_shape))

    for i in range(n_layers):
        model.add(Sequential(layers=[
            Dense(units=unit_sizes[i], activation=None),
            Antirectifier(),
            BatchNormalization()],
            name="antirectifier_block_{}".format(i))
        )

    if flatten:
      model.add(Flatten())
    else:
      model.add(GlobalMaxPooling1D())

    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation=None, name='classification_head'))

    if logits is False:
        model.add(Activation('softmax', name="softmax"))

    loss = SparseCategoricalCrossentropy(from_logits=True) \
        if logits else CategoricalCrossentropy()

    metrics = ['accuracy']

    optimizer = Adam() #optimizers.SGD()

    model.compile(
        loss=loss, 
        optimizer=optimizer, 
        metrics=metrics,
        # jit_compile=jit_compile
    )

    print(model.summary())
    return model



def antirectifier_cnn_1D(input_shape, 
                         num_classes,
                         model_nm='antirectifier_cnn_1D',
                         n_layers=4,
                         filter_sizes=[64, 128, 256, 512],
                         kernel_sizes=[3, 3, 5, 12],
                         dropout_rate=0.2,
                         penultimate_units=128,
                         flatten=True,
                         padding='same',
                         logits=False,
                         max_layer_depth_pool=8,
                         jit_compile=False):

    assert len(filter_sizes) == len(kernel_sizes) == n_layers

    model = Sequential(name=model_nm)
    model.add(Input(input_shape))

    for i in range(n_layers):
        model.add(Sequential(layers=[
            Conv1D(
                filters=filter_sizes[i], kernel_size=kernel_sizes[i], 
                padding=padding, activation=None),
            Antirectifier(),
            BatchNormalization()],
            name="antirectifier_block_{}".format(i))
        )
        if i % 2 == 0 and i < max_layer_depth_pool:
            rate = dropout_rate[i] if hasattr(dropout_rate, '__len__') else dropout_rate
            model.add(MaxPooling1D())
            model.add(Dropout(rate))

    if flatten:
      model.add(Flatten())
    else:
      model.add(GlobalMaxPooling1D())

    model.add(Dense(penultimate_units))
    model.add(Dense(num_classes, activation=None, name='classification_head'))

    if logits is False:
        model.add(Activation('softmax', name="softmax"))

    loss = SparseCategoricalCrossentropy(from_logits=True) \
        if logits else CategoricalCrossentropy()

    metrics = ['accuracy']

    optimizer = Adam() #optimizers.SGD()

    model.compile(
        loss=loss, 
        optimizer=optimizer, 
        metrics=metrics,
        # jit_compile=jit_compile
    )

    print(model.summary())
    return model



def antirectifier_cnn_2D(input_shape, 
                         num_classes,
                         model_nm='antirectifier_cnn_1D',
                         n_layers=4,
                         filter_sizes=[64, 64, 128, 256],
                         kernel_sizes=[3, 3, 5, 8],
                         dropout_rate=0.2,
                         penultimate_units=128,
                         flatten=True,
                         padding='same',
                         logits=False,
                         max_layer_depth_pool=8,
                         jit_compile=False):

    assert len(filter_sizes) == len(kernel_sizes) == n_layers

    model = Sequential(name=model_nm)
    model.add(Input(input_shape))

    for i in range(n_layers):
        model.add(Sequential(layers=[
            Conv2D(
                filters=filter_sizes[i], kernel_size=kernel_sizes[i], 
                padding=padding, activation=None),
            Antirectifier(),
            BatchNormalization()],
            name="antirectifier_block_{}".format(i))
        )
        if i % 2 == 0 and i < max_layer_depth_pool:
            rate = dropout_rate[i] if hasattr(dropout_rate, '__len__') else dropout_rate
            model.add(MaxPooling2D())
            model.add(Dropout(rate))

    if flatten:
      model.add(Flatten())
    else:
      model.add(GlobalMaxPooling2D())

    model.add(Dense(penultimate_units))
    model.add(Dense(num_classes, activation=None, name='classification_head'))

    if logits is False:
        model.add(Activation('softmax', name="softmax"))

    loss = SparseCategoricalCrossentropy(from_logits=True) \
        if logits else CategoricalCrossentropy()

    metrics = ['accuracy']

    optimizer = Adam() #optimizers.SGD()

    model.compile(
        loss=loss, 
        optimizer=optimizer, 
        metrics=metrics,
        jit_compile=jit_compile
    )

    print(model.summary())
    return model



dense_configs = dict(
  antirect_tiny=dict(
    n_layers=2,
    unit_sizes=[32, 64],
    dropout_rate=0.2,
    flatten=True,
    logits=True
  ),
  antirect_small=dict(
    n_layers=4,
    unit_sizes=[64, 128, 256, 512],
    dropout_rate=0.2,
    flatten=True,
    padding='same',
    logits=True
  ),
  antirect_base=dict(
    n_layers=6,
    unit_sizes=[32, 64, 64, 128, 256, 256],
    dropout_rate=0.2,
    flatten=True,
    padding='same',
    logits=True
  ),
  antirect_large=dict(
    n_layers=8,
    unit_sizes=[64, 64, 128, 128, 256, 256, 384, 384],
    dropout_rate=0.2,
    flatten=True,
    padding='same',
    logits=True
  ),
  antirect_xlarge=dict(
    n_layers=10,
    unit_sizes=[64, 64, 128, 128, 256, 256, 384, 384, 512, 512],
    dropout_rate=0.2,
    flatten=True,
    padding='same',
    logits=True
  ),  
)


cnn_configs = dict(
  antirect_tiny=dict(
    n_layers=2,
    filter_sizes=[32, 64],
    kernel_sizes=[3, 3],
    dropout_rate=0.2,
    penultimate_units=32,
    flatten=False,
    padding='same',
    logits=True
  ),
  antirect_small=dict(
    n_layers=4,
    filter_sizes=[64, 128, 256, 512],
    kernel_sizes=[3, 3, 5, 12],
    dropout_rate=0.2,
    penultimate_units=64,
    flatten=False,
    padding='same',
    logits=True
  ),
  antirect_base=dict(
    n_layers=6,
    filter_sizes=[32, 64, 64, 128, 256, 256],
    kernel_sizes=[3, 3, 5, 8, 10, 12],
    dropout_rate=0.2,
    penultimate_units=128,
    flatten=False,
    padding='same',
    logits=True
  ),
  antirect_large=dict(
    n_layers=8,
    filter_sizes=[64, 64, 128, 128, 256, 256, 384, 384],
    kernel_sizes=[3, 3, 5, 8, 10, 12, 12, 16],
    dropout_rate=0.2,
    penultimate_units=128,
    flatten=False,
    padding='same',
    logits=True
  ),
  antirect_xlarge=dict(
    n_layers=10,
    filter_sizes=[64, 64, 128, 128, 256, 256, 384, 384, 512, 512],
    kernel_sizes=[3, 3, 5, 5, 7, 7, 9, 10, 10, 10],
    dropout_rate=0.2,
    penultimate_units=256,
    flatten=False,
    padding='same',
    logits=True
  ),  
)


def build_antirectifier_dense(input_shape, num_classes,                
                              model_nm='antirect_base'):
  cfg = dense_configs[model_nm]

  model = antirectifier_dense(
    input_shape=input_shape,
    num_classes=num_classes,
    n_layers=cfg['n_layers'],
    unit_sizes=cfg['unit_sizes'],
    dropout_rate=cfg['dropout_rate'],
    flatten=cfg['flatten'],
    logits=cfg['logits'],
    model_nm=model_nm
  )
  return model


def build_antirectifier_cnn_1D(input_shape, num_classes, model_nm='antirect_base'):
  cfg = cnn_configs[model_nm]

  model = antirectifier_cnn_1D(
    input_shape=input_shape, num_classes=num_classes,
    n_layers=cfg['n_layers'],
    filter_sizes=cfg['filter_sizes'],
    kernel_sizes=cfg['kernel_sizes'],
    dropout_rate=cfg['dropout_rate'],
    penultimate_units=cfg['penultimate_units'],
    flatten=cfg['flatten'],
    padding=cfg['padding'],
    logits=cfg['logits'],
    model_nm=model_nm
  )
  return model


def build_antirectifier_cnn_2D(input_shape, num_classes, model_nm='antirect_base'):
  cfg = cnn_configs[model_nm]

  model = antirectifier_cnn_2D(
    input_shape=input_shape, num_classes=num_classes,
    n_layers=cfg['n_layers'],
    filter_sizes=cfg['filter_sizes'],
    kernel_sizes=cfg['kernel_sizes'],
    dropout_rate=cfg['dropout_rate'],
    penultimate_units=cfg['penultimate_units'],
    flatten=cfg['flatten'],
    padding=cfg['padding'],
    logits=cfg['logits'],
    model_nm=model_nm
  )
  return model



if False:
  m1 = build_antirectifier_cnn_1D(input_shape[:-1], num_classes)
  m2 = build_antirectifier_cnn_2D(input_shape, num_classes)



