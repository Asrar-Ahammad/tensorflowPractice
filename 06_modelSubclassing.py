import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist

print(tf.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0


class CNNBLOCK(layers.Layer):
    def __init__(self, out_channels, kernel_size=5):
        super(CNNBLOCK, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


model = keras.Sequential(
    [
        CNNBLOCK(32),
        CNNBLOCK(64),
        CNNBLOCK(128),
        layers.Flatten(),
        layers.Dense(10)
    ]
)


class ResBlock(layers.Layer):
    def __int__(self, channels):
        super(ResBlock, self).__init__()
        self.cnn1 = CNNBLOCK(channels[0])
        self.cnn2 = CNNBLOCK(channels[1])
        self.cnn3 = CNNBLOCK(channels[2])
        self.pooling = layers.MaxPooling2D()
        self.identity_mapping = layers.Conv2D(channels[1], 1, padding='same')

    def call(self, input_tensor, training=False):
        x = self.cnn1(input_tensor, training=training)
        x = self.cnn2(x, training=training)
        x = self.cnn3(x + self.identity_mapping(input_tensor), training=training)
        return self.pooling(x)


class Res_like(keras.Model):
    def __init__(self, num_classes=10):
        super(Res_like, self).__init__()
        self.block1 = ResBlock([32, 32, 64])
        self.block2 = ResBlock([128, 128, 256])
        self.block3 = ResBlock([128, 256, 512])
        self.pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    def call(self, input_tensor, training=False):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.pool(x)
        return self.classifier(x)


model = Res_like(num_classes=10)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.legacy.Adam(),
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=2)
print(model.summary())
model.evaluate(x_test, y_test, verbose=2, batch_size=64)
