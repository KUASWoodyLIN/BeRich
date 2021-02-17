import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


def load_dnn_model_v1(weight_path=None):
    inputs = keras.Input((12,))
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs, name='model')
    if weight_path:
        model.load_weights(weight_path)
    return model


def load_dnn_model_v2(weight_path=None):
    inputs_1 = keras.Input((12,))
    inputs_2 = keras.Input((15,))

    x1 = keras.layers.Dense(64, activation='relu')(inputs_1)
    x1 = keras.layers.Dropout(0.3)(x1)
    x1 = keras.layers.Dense(64, activation='relu')(x1)

    x2 = keras.layers.Dense(64, activation='relu')(inputs_2)
    x2 = keras.layers.Dropout(0.3)(x2)
    x2 = keras.layers.Dense(64, activation='relu')(x2)

    x = keras.layers.Concatenate()([x1, x2])
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model([inputs_1, inputs_2], outputs, name='model')
    if weight_path:
        model.load_weights(weight_path)
    return model


def load_dnn_model_v3(weight_path=None):
    inputs_1 = keras.Input((14,))
    inputs_2 = keras.Input((15,))

    x1 = keras.layers.Dense(64, activation='relu')(inputs_1)
    x1 = keras.layers.Dropout(0.3)(x1)
    x1 = keras.layers.Dense(64, activation='relu')(x1)

    x2 = keras.layers.Dense(64, activation='relu')(inputs_2)
    x2 = keras.layers.Dropout(0.3)(x2)
    x2 = keras.layers.Dense(64, activation='relu')(x2)

    x = keras.layers.Concatenate()([x1, x2])
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model([inputs_1, inputs_2], outputs, name='model')
    if weight_path:
        model.load_weights(weight_path)
    return model


def load_dnn_model_v4(weight_path=None):
    inputs_1 = keras.Input((14,))
    inputs_2 = keras.Input((5,))

    x1 = keras.layers.Dense(64, activation='relu')(inputs_1)
    x1 = keras.layers.Dropout(0.3)(x1)
    x1 = keras.layers.Dense(64, activation='relu')(x1)

    x2 = keras.layers.Dense(64, activation='relu')(inputs_2)
    x2 = keras.layers.Dropout(0.3)(x2)
    x2 = keras.layers.Dense(64, activation='relu')(x2)

    x = keras.layers.Concatenate()([x1, x2])
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model([inputs_1, inputs_2], outputs, name='model')
    if weight_path:
        model.load_weights(weight_path)
    return model


def load_dnn_model_v5(weight_path=None):
    inputs_1 = keras.Input((15,))
    inputs_2 = keras.Input((5,))

    x1 = keras.layers.Dense(64, activation='relu')(inputs_1)
    x1 = keras.layers.Dropout(0.3)(x1)
    x1 = keras.layers.Dense(64, activation='relu')(x1)

    x2 = keras.layers.Dense(64, activation='relu')(inputs_2)
    x2 = keras.layers.Dropout(0.3)(x2)
    x2 = keras.layers.Dense(64, activation='relu')(x2)

    x = keras.layers.Concatenate()([x1, x2])
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model([inputs_1, inputs_2], outputs, name='model')
    if weight_path:
        model.load_weights(weight_path)
    return model


def load_dnn_model_v6(weight_path=None):
    inputs_1 = keras.Input((8,))
    inputs_2 = keras.Input((4,))
    inputs_3 = keras.Input((3,))
    inputs_4 = keras.Input((4,))
    inputs_5 = keras.Input((5,))

    # Input 1
    x1 = keras.layers.Dense(64, activation='relu')(inputs_1)
    x1 = keras.layers.Dense(64, activation='relu')(x1)
    # x1 = keras.layers.Dropout(0.3)(x1)
    x1 = keras.layers.Dense(64, activation='relu')(x1)
    # Input 2
    x2 = keras.layers.Dense(64, activation='relu')(inputs_2)
    x2 = keras.layers.Dense(64, activation='relu')(x2)
    # x2 = keras.layers.Dropout(0.3)(x2)
    x2 = keras.layers.Dense(64, activation='relu')(x2)
    # Input 3
    x3 = keras.layers.Dense(64, activation='relu')(inputs_3)
    x3 = keras.layers.Dense(64, activation='relu')(x3)
    # x3 = keras.layers.Dropout(0.3)(x3)
    x3 = keras.layers.Dense(64, activation='relu')(x3)
    # Input 4
    x4 = keras.layers.Dense(64, activation='relu')(inputs_4)
    x4 = keras.layers.Dense(64, activation='relu')(x4)
    # x4 = keras.layers.Dropout(0.3)(x4)
    x4 = keras.layers.Dense(64, activation='relu')(x4)
    # Input 5
    x5 = keras.layers.Dense(64, activation='relu')(inputs_5)
    x5 = keras.layers.Dense(64, activation='relu')(x5)
    # x5 = keras.layers.Dropout(0.3)(x5)
    x5 = keras.layers.Dense(64, activation='relu')(x5)
    # Combine
    x = keras.layers.Concatenate()([x1, x2, x3, x4, x5])
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model([inputs_1, inputs_2, inputs_3, inputs_4, inputs_5], outputs, name='model')
    if weight_path:
        model.load_weights(weight_path)
    return model


def load_dnn_model_v7(weight_path=None):
    inputs_1 = keras.Input((8,))
    inputs_2 = keras.Input((4,))
    inputs_3 = keras.Input((3,))
    inputs_4 = keras.Input((4,))
    inputs_5 = keras.Input((5,))
    inputs_6 = keras.Input((5,))

    # Input 1
    x1 = keras.layers.Dense(64, activation='relu')(inputs_1)
    x1 = keras.layers.Dense(64, activation='relu')(x1)
    # x1 = keras.layers.Dropout(0.3)(x1)
    x1 = keras.layers.Dense(64, activation='relu')(x1)
    # Input 2
    x2 = keras.layers.Dense(64, activation='relu')(inputs_2)
    x2 = keras.layers.Dense(64, activation='relu')(x2)
    # x2 = keras.layers.Dropout(0.3)(x2)
    x2 = keras.layers.Dense(64, activation='relu')(x2)
    # Input 3
    x3 = keras.layers.Dense(64, activation='relu')(inputs_3)
    x3 = keras.layers.Dense(64, activation='relu')(x3)
    # x3 = keras.layers.Dropout(0.3)(x3)
    x3 = keras.layers.Dense(64, activation='relu')(x3)
    # Input 4
    x4 = keras.layers.Dense(64, activation='relu')(inputs_4)
    x4 = keras.layers.Dense(64, activation='relu')(x4)
    # x4 = keras.layers.Dropout(0.3)(x4)
    x4 = keras.layers.Dense(64, activation='relu')(x4)
    # Input 5
    x5 = keras.layers.Dense(64, activation='relu')(inputs_5)
    x5 = keras.layers.Dense(64, activation='relu')(x5)
    # x5 = keras.layers.Dropout(0.3)(x5)
    x5 = keras.layers.Dense(64, activation='relu')(x5)
    # Input 6
    x6 = keras.layers.Dense(64, activation='relu')(inputs_6)
    x6 = keras.layers.Dense(64, activation='relu')(x6)
    # x6 = keras.layers.Dropout(0.3)(x6)
    x6 = keras.layers.Dense(64, activation='relu')(x6)
    # Combine
    x = keras.layers.Concatenate()([x1, x2, x3, x4, x5, x6])
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model([inputs_1, inputs_2, inputs_3, inputs_4, inputs_5, inputs_6], outputs, name='model')
    if weight_path:
        model.load_weights(weight_path)
    return model
