"""
Model definition.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def make_model(param):
    """This function defines the 3d convolutional neural network model."""
    # Add categorical input type
    if param['cat_input_type'] == 'None':
        condition = lambda x: x*np.zeros(10)
        # condition = lambda x: x*0
    elif param['cat_input_type'] == 'sex':
        condition = lambda x: x*np.concatenate((np.ones(2),np.zeros(8)))
        # condition = lambda x: x[:,:2]
    elif param['cat_input_type'] == 'study':
        condition = lambda x: x*np.concatenate((np.zeros(2),np.ones(8)))
        # condition = lambda x: x[:,2:]
    elif param['cat_input_type'] == 'sex_study':
        # mask = np.ones(10)
        # condition = lambda x: x
        condition = lambda x: x
    else:
        raise Exception('Categorical input type selected is undefined')

    initializer = lambda n_chan: tf.keras.initializers.he_normal()
    n_chan = 8

    # n-Maps definition
    map_models = dict()
    models_outputs = []
    models_inputs = []
    for k in range(param['n_maps']):
        map_models[str(k)] = tf.keras.Sequential()
        map_models[str(k)].add(tf.keras.Input(shape=(91,109,55)))
        map_models[str(k)].add(layers.Reshape((91,109,55,1)))
        models_inputs.append(map_models[str(k)].input)
        models_outputs.append(map_models[str(k)].output)

    x = tf.keras.layers.concatenate(models_outputs)

    if param['arc_type'] == 1:
        map_shape = x.get_shape()
        model_cat = tf.keras.Sequential()
        model_cat.add(tf.keras.Input(shape=10))
        model_cat.add(layers.Lambda(condition))
        model_cat.add(layers.Dense(np.prod(map_shape[1:])))
        model_cat.add(layers.Reshape(map_shape[1:]))
        x = layers.Add()([x,model_cat.output])

    for layer in range(5):
        x = layers.Conv3DTranspose(2**layer*n_chan, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                    kernel_initializer=initializer(n_chan), use_bias=False)(x)
        x = layers.ReLU()(x)
        x = layers.Conv3DTranspose(2**layer*n_chan, (3, 3, 3), strides=(1, 1, 1), padding='same',
                                    kernel_initializer=initializer(n_chan), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    concat_layer_shape = x.get_shape()# model.layers
    if param['arc_type'] == 1:
        x = layers.Reshape((np.prod(concat_layer_shape[1:4])*(concat_layer_shape[-1]),))(x)
        x = layers.Dense(640, activation="relu")(x) # try activation="softplus"
        x = layers.Dense(100, activation="relu")(x) # try activation="softplus"
    elif param['arc_type'] == 2:
        x = layers.Reshape((np.prod(concat_layer_shape[1:4])*(concat_layer_shape[-1]),))(x)
        model = tf.keras.models.Model(inputs=models_inputs, outputs=x)

        model_cat = tf.keras.Sequential()
        model_cat.add(tf.keras.Input(shape=10))
        model_cat.add(layers.Lambda(condition))

        y = layers.concatenate([model.output, model_cat.output])
        x = layers.Dense(640, activation="relu")(y) # try activation="softplus"
        x = layers.Dense(100, activation="relu")(x) # try activation="softplus"
    elif param['arc_type'] == 3:
        x = layers.Reshape((np.prod(concat_layer_shape[1:4])*(concat_layer_shape[-1]),))(x)
        x = layers.Dense(640, activation="relu")(x) # try activation="softplus"
        model = tf.keras.models.Model(inputs=models_inputs, outputs=x)

        model_cat = tf.keras.Sequential()
        model_cat.add(tf.keras.Input(shape=10))
        model_cat.add(layers.Lambda(condition))

        y = layers.concatenate([model.output, model_cat.output])
        x = layers.Dense(100, activation="relu")(y) # try activation="softplus"
    elif param['arc_type'] == 4:
        x = layers.Reshape((np.prod(concat_layer_shape[1:4])*(concat_layer_shape[-1]),))(x)
        x = layers.Dense(640, activation="relu")(x) # try activation="softplus"
        x = layers.Dense(100, activation="relu")(x) # try activation="softplus"
        model = tf.keras.models.Model(inputs=models_inputs, outputs=x)

        model_cat = tf.keras.Sequential()
        model_cat.add(tf.keras.Input(shape=10))
        model_cat.add(layers.Lambda(condition))

        x = tf.keras.layers.concatenate([model.output, model_cat.output])
    else:
        raise Exception('Architecture type selected is undefined')

    x = layers.Dense(1, activation="linear")(x)
    models_inputs.append(model_cat.input)
    final_model = tf.keras.models.Model(inputs=models_inputs, outputs=x)
    final_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=param['lr'], beta_1=0.9,
                                        beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                        name='Adam'),
    loss = tf.keras.losses.MeanAbsoluteError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.MeanAbsoluteError()])
    return final_model
