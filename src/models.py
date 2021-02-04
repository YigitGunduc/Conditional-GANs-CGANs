import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
import numpy as np

WIDTH, HEIGHT = 28, 28
num_classes = 10
img_channel = 1
img_shape = (WIDTH, HEIGHT, img_channel)
noise_dim = 100

def build_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(7*7*256, use_bias=False,input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    z = layers.Input(shape=(noise_dim,))
    label = layers.Input(shape=(1,))
 
    label_embedding = layers.Embedding(num_classes, noise_dim, input_length = 1)(label)
    label_embedding = layers.Flatten()(label_embedding)
    joined = layers.multiply([z, label_embedding])

    img = model(joined)
    return Model([z, label], img)

def build_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                      input_shape=[28, 28, 2]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
 
    img = layers.Input(shape=(img_shape))
    label = layers.Input(shape=(1,))
 
    label_embedding = layers.Embedding(input_dim=num_classes, output_dim=np.prod(img_shape), input_length = 1)(label)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape(img_shape)(label_embedding)
 
    concat = layers.Concatenate(axis=-1)([img, label_embedding])
    prediction = model(concat)
    return Model([img, label], prediction)

