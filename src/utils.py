import numpy as np
from matplotlib import pyplot as plt

def normalize(train, test):
  # convert from integers to floats
  train_norm = train.astype('float32')
  test_norm = test.astype('float32')
  # normalize to range 0-1
  train_norm = train_norm / 255.0
  test_norm = test_norm / 255.0
  # return normalized images
  return train_norm, test_norm

def display_sample(X, y, index):
  print(y[index])
  IMG = X[index]
  IMG = IMG.reshape(28,28)
  plt.imshow(IMG)

def check_cuda():
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        return False 
    return True
