import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Importing dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape)

# Normalization and Type conversion
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


# Building model
# model = keras.Sequential([
#     keras.Input(shape=(32, 32, 3)),
#     layers.Conv2D(32, 3, padding='valid', activation='relu'),  # Valid: default ; same:option
#     layers.MaxPooling2D(pool_size=(2, 2)),
#     layers.Conv2D(64, 3, activation='relu'),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#     layers.Conv2D(128, 3, activation='relu'),
#     layers.Flatten(),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(10)
# ])

# Functional API
def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding='same',kernel_regularizer = regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)  # It also acts as a regularizing effect.
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 5, padding='same', kernel_regularizer = regularizers.l2(0.01))(x)  # 5 is kernel size
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3, padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = my_model()
# If softmax activation is not given in output layer we use from_logits = false.
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)

print(model.summary())
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=2)
model.evaluate(X_test, y_test, batch_size=64, verbose=2)
