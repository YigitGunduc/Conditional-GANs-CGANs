import tensorflow as tf 
import sys
from tensorflow.keras.models import load_model
import numpy as np 
from matplotlib import pyplot as plt

generator = load_model('generator')

def generate(model, label):

    name2idx = {'T-shirt/top' : 0,
                'Trouser' : 1,
                'Pullover' : 2,
                'Dress' : 3,
                'Coat' : 4,
                'Sandal' : 5,
                'Shirt' : 6, 
                'Sneaker' : 7,
                'Bag' : 8, 
                'Ankle boot' : 9}
    
    label = [name2idx[label]]

    label = np.expand_dims(label, axis=1)

    noise = np.random.normal(0,1,size = (1, 100))

    img = np.array(model([noise,label]))
    plt.imshow(img.reshape(28,28))

generate(generator, sys.args[1])
