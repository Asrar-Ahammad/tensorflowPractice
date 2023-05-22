import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist
print(tf.__version__)

(x_train, y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0












