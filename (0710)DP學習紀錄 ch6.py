import numpy as np									#numpy model 為python程式庫，支援維度與陣列運算
import pandas as pd
from keras.utils import np_utils					#import keras.utils，因為之後要把label轉成One-hot encoding
np.random.seed(10)									#set seed 使每次隨機產生之資料有相同輸出

from keras.datasets import mnist
(X_train_image, y_train_label), \
(X_test_image, y_test_label) = mnist.load_data()

print('train data=', len(X_train_image))
print(' test data=', len(X_test_image))
print('X_train_image:',X_train_image.shape)
print('y_train_image:',y_train_label.shape)

import matplotlib.pyplot as plt
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image, cmap='binary')
    plt.show()
plot_image(X_train_image[0])
y_train_label[0]

import matplotlib.pyplot as plt
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25:
        num=25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title = "label=" + str(labels[idx])
        if len(prediction)>0:
            title+=", predict=" +str(prediction[idx])
            
        ax.set_title(title, fontsize = 10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
plot_images_labels_prediction(X_train_image, y_train_label, [], 0, 10)
print('X_test_image:', X_test_image.shape)
print('y_test_label:', y_test_label.shape)
plot_images_labels_prediction(X_test_image, y_test_label, [], 0, 10)

x_Train = X_train_image.reshape(60000, 784).astype('float32')
x_Test = X_test_image.reshape(10000, 784).astype('float32')
X_train_image[0]
x_Train_normalize = x_Train/255
x_Test_normalize = x_Test/255
x_Train_normalize[0]

y_train_label[:5]
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)
y_TrainOneHot[:5]
