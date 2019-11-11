from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 加载tensorflow官方提供数据集fashion MIIST
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 将训练样本和测试样本的data reshape成28 x 28 x 1 大小的数据，因为数据集是灰度图像，如果不reshape的话会少一个维度
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# 将像素值归一化到0~1
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# 创建模型
model = models.Sequential()
# 添加一个卷基层，卷积核大小为3x3,卷积核深度为32,激活函数为relu，输入数据大小为28x28x1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 添加一个2x2的池化层
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))

# 显示模型现在的结构
model.summary()

# 将最后一层卷基层拉伸成一个向量
model.add(layers.Flatten())
# 添加一个全连接层，正则化项为L2正则大小为0.001，激活函数为relu
model.add(layers.Dense(128,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
# 添加20%的Dropout
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(layers.Dropout(0.2))
# 最后用softmax输出10个类别
model.add(layers.Dense(10, activation='softmax'))

# 输出当前模型架构
model.summary()

# 编译模型，优化函数用adam，损失函数用sparse_categorical_crossentropy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型,进行10次迭代，batch_size=32
history = model.fit(train_images, train_labels, epochs=10, batch_size=32,
                    validation_data=(test_images, test_labels))
# history = model.fit(train_images, train_labels, epochs=10)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
plt.show()
