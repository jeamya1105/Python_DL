"""[Data 預先處理]"""
#import model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)
#read mnist data
(x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()
#features turn to 4 dimension & normalize
x_Train4D = x_Train.reshape(x_Train.shape[0], 28, 28, 1).astype('float32')		#width, height, color
x_Test4D = x_Test.reshape(x_Test.shape[0], 28, 28, 1).astype('float32')
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255
#label encoding by Onehot
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)

"""[Build Model]"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
model = Sequential()
#Convolution layer 1
model.add(Conv2D(filters = 16,				#random 16 filter weight
                 kernel_size=(5, 5),		#filter size 5*5
                 padding = 'same',			#convolution, image size unchanged
                 input_shape = (28, 28, 1), #total 3 dimension, 1&2 image size, 3 single color(gray scale)
                 activation = 'relu'))		#function
#Pooling layer 1
model.add(MaxPooling2D(pool_size=(2, 2)))	#28/2, 28/2 -> 14*14
#Convolution layer 2
model.add(Conv2D(filters = 36,
                 kernel_size=(5, 5),
                 padding = 'same',
                 activation = 'relu'))
#Pooling layer 2
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#Flatten layer
model.add(Flatten())						#36*7*7 float number = 1764 neurons
#Hidden layer
model.add(Dense(128, activation = 'relu'))	#128 neurons
model.add(Dropout(0.5))
#Output layer
model.add(Dense(10, activation = 'softmax'))
print(model.summary())

"""[Start]"""
#Compile
model.compile(loss = 'categorical_crossentropy',		#loss function
              optimizer = 'adam',						#optimize
              metrics = ['accuracy'])					#evaluate use accuracy
#Train
train_history = model.fit(x = x_Train4D_normalize,		#features
                          y = y_TrainOneHot,			#label
                          validation_split = 0.2,		#train 80%  validate 20%
                          epochs = 10,
                          batch_size = 300,
                          verbose = 2)					#顯示，0為不顯示，1為每個epochs都輸出有進度條，2為每個epochs都輸出沒有進度條
#Graph
import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')
"""[Evaluate & Prediction]"""
#Evaluate accuracy
scores = model.evaluate(x_Test4D_normalize, y_TestOneHot)	#features, label
scores[1]
#Prediction
prediction = model.predict_classes(x_Test4D_normalize)
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
plot_images_labels_prediction(x_Test, y_Test, prediction, idx = 0)		#test data, label, predict, 0~9共10筆資料

"""[Confusion matrix]"""
#顯示表格
import pandas as pd
pd.crosstab(y_Test, prediction, rownames=['label'], colnames=['predict'])

