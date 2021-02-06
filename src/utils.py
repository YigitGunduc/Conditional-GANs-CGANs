import numpy as np

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
    '''
    :param: X dataset containing the images
    :param: y dataset containing the labels
    :param: index images and labels point in the
    dataset if index == 'random' function will pick randomly
    
    displays the image and the label on the given index
    '''
    import matplotlib.pyplot as plt

    if index == 'random':
        import random 
        index = random.randint(0, len(X))

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
