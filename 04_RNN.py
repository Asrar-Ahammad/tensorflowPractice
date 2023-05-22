import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Importing dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)

# Normalization
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Building model
# Simple Recurent Neural Network
model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(layers.SimpleRNN(256, return_sequences=True, activation='relu'))  # For RNN we use tanh activation
model.add(layers.SimpleRNN(256, activation='relu'))
model.add(layers.Dense(10))

# Gated Recurent Unit
model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(layers.GRU(256, return_sequences=True, activation='relu'))  # For RNN we use tanh activation
model.add(layers.GRU(256, activation='relu'))
model.add(layers.Dense(10))

# Long Short Term Memory
model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(layers.LSTM(256, return_sequences=True, activation='relu'))  # For RNN we use tanh activation
model.add(layers.LSTM(256, activation='relu'))
model.add(layers.Dense(10))

# Bidirectional LSTM
model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(layers.Bidirectional(
    layers.LSTM(256, return_sequences=True, activation='relu')))  # For RNN, we use tanh activation
model.add(layers.Bidirectional(
    layers.SimpleRNN(256, activation='relu')))
model.add(layers.Dense(10))
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)

model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(X_test, y_test, batch_size=64, verbose=2)
