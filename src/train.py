import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist, mnist
from models import build_generator_model, build_discriminator_model
import utils


WIDTH, HEIGHT = 28, 28
num_classes = 10
img_channel = 1
img_shape = (WIDTH, HEIGHT, img_channel)
noise_dim = 100
dataset = 'fashion_mnist'
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
learning_rate = 1e-4
Lamda = 10
epochs = 50000


if dataset == 'mnist':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
if dataset == 'fashion_mnist':
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
else:
    raise RuntimeError('Dataset not found')

X_train, X_test = utils.normalize(X_train, X_test)

utils.display_sample(X_train, y_train, index='random')


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


discriminator = build_discriminator_model()
generator = build_generator_model()

generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)


def save_models(epochs, learning_rate):
    generator.save(f'generator-epochs-{epochs}-learning_rate-{learning_rate}.h5')
    discriminator.save(f'discriminator-epochs-{epochs}-learning_rate-{learning_rate}.h5')


tf.function
def train_step(batch_size=512):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    Xtrain, labels = X_train[idx], y_train[idx]
    with tf.device('/device:GPU:0' if utils.check_cuda else '/cpu:0'):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      
            z = np.random.normal(0, 1, size=(batch_size, noise_dim))

            generated_images = generator([z, labels], training=True)
            real_output = discriminator([Xtrain, labels], training=True)
            fake_output = discriminator([generated_images, labels], training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        tf.print(f'Genrator loss: {gen_loss} Discriminator loss: {disc_loss}')
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(epochs):
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        train_step()

        if epoch % 1000 == 0:
            save_models(epoch, learning_rate)

train(epochs)
