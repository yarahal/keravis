import numpy as np
from math import sqrt, log
import tensorflow as tf


def pixel_scaling(dim):
    return log(dim)


def find_closest_factors(n):
    '''
    Efficient algorithm for finding the closest two factors of a number n
    '''
    x = int(sqrt(n))
    y = n/x
    while y != int(y):
        x -= 1
        y = n/x
    return x, int(y)


def normalize(img):
    '''
    Image minmax normalization
    '''
    img -= np.min(img, axis=(0, 1))
    img /= np.max(img, axis=(0, 1)) + 1e-7


def clone_function_1(layer):
    layer_config = layer.get_config()
    if 'activation' in layer_config and (layer_config['activation'] == 'sigmoid' or layer_config['activation'] == 'softmax'):
        layer_config['activation'] = 'linear'
    new_layer = type(layer).from_config(layer_config)
    return new_layer


def clone_function_2(layer):
    if isinstance(layer, tf.keras.Model):
        return tf.keras.models.clone_model(layer, layer.input, clone_function_2)
    layer_config = layer.get_config()
    if 'activation' in layer_config and layer_config['activation'] == 'relu':
        layer_config['activation'] = guided_relu
    elif 'activation' in layer_config and (layer_config['activation'] == 'sigmoid' or layer_config['activation'] == 'softmax'):
        layer_config['activation'] = guided_linear
    new_layer = type(layer).from_config(layer_config)
    return new_layer


@tf.custom_gradient
def guided_relu(x):
    y = tf.nn.relu(x)

    def grad(upstream):
        dx = tf.where(x < 0., 0., 1.)
        return tf.maximum(upstream * dx, 0)
    return y, grad


@tf.custom_gradient
def guided_linear(x):
    y = x

    def grad(upstream):
        return tf.maximum(upstream, 0)
    return y, grad
