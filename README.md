# Generic Tensorflow Training Module

## Installation
```{bash}
git clone https://github.com/ifrit98/train_generic.git
cd train_generic
pip install .
```

## Basic Usage (complete code example)
```{python}
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Antirectifier(layers.Layer):
    def __init__(self, initializer="he_normal", **kwargs):
        super(Antirectifier, self).__init__(**kwargs)
        self.initializer = keras.initializers.get(initializer)

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(output_dim * 2, output_dim),
            initializer=self.initializer,
            name="kernel",
            trainable=True,
        )

    def call(self, inputs):
        inputs -= tf.reduce_mean(inputs, axis=-1, keepdims=True)
        pos = tf.nn.relu(inputs)
        neg = tf.nn.relu(-inputs)
        concatenated = tf.concat([pos, neg], axis=-1)
        mixed = tf.matmul(concatenated, self.kernel)
        return mixed

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(Antirectifier, self).get_config()
        config = {"initializer": keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))


def test_model(input_shape=(784,)):# Build the model
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(256),
            Antirectifier(),
            layers.Dense(256),
            Antirectifier(),
            layers.Dropout(0.5),
            layers.Dense(10),
        ]
    )

    # Compile the model
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    print(model.summary())
    return model



def demo():

    from train_generic import mnist

    FLAGS = {
        'model_cfg': {
            'model_src_file': 'C:\\Users\\stgeorge\\Desktop\\blackriver_projects\\FRONTROW\\train_generic\\train_generic\\test.py', 
            'model_fn_name': 'test_model',  #name of callable (for pseudo-metaprogramming)
            'model_fn_args': {'input_shape': (784,)}
        },

        'train_cfg': {
            'epochs': 10, 'num_classes': 10, 'redirect_stdout': False, 'monitor': 'loss',
            'plot_model': True,
            'labels': [str(i) for i in range(10)],
        },

        'data_cfg': {
            'data_loader_file': 'C:\\Users\\stgeorge\\Desktop\\blackriver_projects\\FRONTROW\\train_generic\\data\\data_loader.py',
            'data_loader_fn_name': 'mnist',
            'data_loader_args': {
                'batch_size': 128, 'vectorize': True, 'subsample': True, 'take_n': 1000,
                'return_val_set': True
            },
        }
    }


    from train_generic import train
    results = train(
        model_fn=test_model, 
        model_cfg=FLAGS.get('model_cfg', {}),
        data_fn=mnist,
        data_cfg=FLAGS.get('data_cfg', {}),
        **FLAGS.get('train_cfg', {})
    )
    print(results)


    from train_generic import training_run
    training_run(FLAGS=FLAGS)
```