import os

import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
# X_train = X_train.reshape(-1, 28 * 28)  # without normalization
# X_test = X_test.reshape(-1, 28 * 28)

X_train = X_train.reshape(-1,28*28)/255.0 # Normalization increases accuracy
X_test = X_test.reshape(-1,28*28)/255.0
print(X_train.shape)

# Sequential API
# model = keras.Sequential([ # passing layers as list to model.
#     keras.Input(shape=(28*28)),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(10)
# ])
# print(model.summary()) // if we include input layer.

# Sequential API - 2
model = keras.Sequential()
model.add(layers.Input(shape=(28*28)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu',name='my_layer'))
model.add(layers.Dense(10))

model = keras.Model(inputs = model.inputs,
                    # outputs = [model.layers[-1].output]
                    #outputs = [model.get_layer('my_layer').output] # We can access layers by their name.
                    outputs = [layer.output for layer in model.layers])

feature = model.predict(X_train)
for features in feature:
    print(features.shape)
# print('feature shape :',feature.shape)


# Functional API
# input = layers.Input(shape=(28 * 28))
# x = layers.Dense(512, activation='relu',name='first_layer')(input)
# x = layers.Dense(256, activation='relu',name='second_layer')(x)
# output = layers.Dense(10, activation='softmax')(x)
# model = keras.Model(inputs=input, outputs=output)

# import sys
# sys.exit()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Only set it to True if you are not using softmax in model.
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5, verbose=2)
print(model.summary())
model.evaluate(X_test, y_test, verbose=2)
