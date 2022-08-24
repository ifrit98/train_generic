import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.layers import experimental



def mlp(x, hidden_units, dropout_rate):
    """
    ## Implement multilayer perceptron (MLP)
    """
    for units in hidden_units:
        x = keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


class Patches(keras.layers.Layer):
    """
    ## Implement patch creation as a layer
    """
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(keras.layers.Layer):
    """
    ## Implement the patch encoding layer
    The `PatchEncoder` layer will linearly transform a patch by projecting it into a
    vector of size `projection_dim`. In addition, it adds a learnable position
    embedding to the projected vector.
    """
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = keras.layers.Dense(units=projection_dim)
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# CONFIGURATION FOR MODEL SIZES
model_configs = dict(
    vit_tiny={
        'image_size': 72, # vision transformer image reduction size (sqaure)
        'patch_size': 6,
        'transformer_layers': 1,
        'mlp_head_units': [256, 128],
        'num_heads': 2,
        'projection_dim': 64,
        'augment_data': False,
        'mean': 0, 'var': 1,
        'weight_decay': 0.0001,
        'learning_rate': 0.001
    },
    vit_small={
        'image_size': 72, # vision transformer image reduction size (sqaure)
        'patch_size': 6,
        'transformer_layers': 2,
        'mlp_head_units': [512, 256],
        'num_heads': 4,
        'projection_dim': 64,
        'augment_data': False,
        'mean': 0, 'var': 1,
        'weight_decay': 0.0001,
        'learning_rate': 0.001
    },
    vit_base={
        'image_size': 72, # vision transformer image reduction size (sqaure)
        'patch_size': 6,
        'transformer_layers': 4,
        'mlp_head_units': [1024, 512],
        'num_heads': 4,
        'projection_dim': 64,
        'augment_data': False,
        'mean': 0, 'var': 1,
        'weight_decay': 0.0001,
        'learning_rate': 0.001
    },
    vit_large={
        'image_size': 72, # vision transformer image reduction size (sqaure)
        'patch_size': 6,
        'transformer_layers': 8,
        'mlp_head_units': [2056, 1024],
        'num_heads': 4,
        'projection_dim': 64,
        'augment_data': False,
        'mean': 0, 'var': 1,
        'weight_decay': 0.0001,
        'learning_rate': 0.001
    },
    vit_xlarge={
        'image_size': 72, # vision transformer image reduction size (sqaure)
        'patch_size': 6,
        'transformer_layers': 16,
        'mlp_head_units': [2056, 1024],
        'num_heads': 8,
        'projection_dim': 128,
        'augment_data': False,
        'mean': 0, 'var': 1,
        'weight_decay': 0.0001,
        'learning_rate': 0.001
    }
)


def vit(input_shape, num_classes,
        transformer_layers=8,
        num_heads=4, augment_data=False,
        mlp_head_units=[2056,1024],
        projection_dim=64,
        image_size=72, # resize input to (n x n)
        patch_size=6,
        mean=0.0, var=1.0,
        weight_decay=0.0001,
        learning_rate=0.001):

    print("Model initialized with data mean {} and var {}".format(mean, var))

    num_patches = (image_size // patch_size) ** 2
    transformer_units = [projection_dim * 2, projection_dim]

    if augment_data:
        data_augmentation = keras.Sequential(
            [
                experimental.preprocessing.Normalization(mean=mean, variance=var),
                experimental.preprocessing.Resizing(image_size, image_size),
                experimental.preprocessing.RandomFlip("horizontal"),
                experimental.preprocessing.RandomRotation(factor=0.02),
                experimental.preprocessing.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_augmentation",
        )
    else:
        data_augmentation = keras.Sequential(
            [
                experimental.preprocessing.Normalization(mean=mean, variance=var),
                experimental.preprocessing.Resizing(image_size, image_size),
            ],
            name="data_normalization",
        )

    inputs = keras.layers.Input(shape=input_shape)

    # Augment data.
    augmented = data_augmentation(inputs)

    # Create patches.
    patches = Patches(patch_size)(augmented)

    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = keras.layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = keras.layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = keras.layers.Flatten()(representation)
    representation = keras.layers.Dropout(0.5)(representation)

    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    # Classify outputs.
    logits = keras.layers.Dense(num_classes)(features)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)

    # https://arxiv.org/abs/1711.05101
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )
    return model


def build_vit(input_shape, num_classes, mean=None, var=None, model_nm='vit_base'):
    # Get configuration dict by str name
    cfg = model_configs[model_nm]

    # Construct base model
    net = vit(
        input_shape=input_shape, num_classes=num_classes, 
        transformer_layers=cfg['transformer_layers'],
        num_heads=cfg['num_heads'],
        augment_data=cfg['augment_data'],
        mlp_head_units=cfg['mlp_head_units'],
        projection_dim=cfg['projection_dim'],
        image_size=cfg['image_size'],
        patch_size=cfg['patch_size'],
        mean=cfg['mean'] if mean is None else mean, 
        var=cfg['var'] if var is None else var,
        weight_decay=cfg['weight_decay'],
        learning_rate=cfg['learning_rate']
    ); print(net.summary())
    
    return net