from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


