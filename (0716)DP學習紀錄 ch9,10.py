"""[Download Cifar10 datasets]"""
from keras.datasets import cifar10
import numpy as np
np.random.seed(10)
(x_img_train, y_label_train), \
(x_img_test, y_label_test) = cifar10.load_data()
print('train : ', len(x_img_train))
print('test : ', len(x_img_test))

"""[Datasets info]"""
x_img_train.shape		#output 4 dimension(total data, image length, image width, RGB=3)
x_img_test[0]
y_label_train.shape
#every number 對映 label
label_dict = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
#顯示trainning 前10筆data function
import matplotlib.pyplot as plt
def plot_images_labels_prediction(images, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25:
        num=25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title = str(i) + ',' + label_dict[labels[i][0]]
        if len(prediction)>0:
            title+="=>" + label_dict[prediction[i]]
            
        ax.set_title(title, fontsize = 10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
plot_images_labels_prediction(x_img_train, y_label_train, [], 0)

"""[Data 預先處理]"""
#第一筆資料 第一個pixel info
x_img_train[0][0][0]
#data normalize 使 every pixel RGB value between 0 & 1
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0

y_label_train.shape		#50000 data, every data is a number between 0 and 9
y_label_train[:5]		#可看到5筆資料都是0~9數字, 為影像的分類
#labels transform to OneHot encoding
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
y_label_train_OneHot.shape
y_label_train_OneHot[:5]

"""[Build model]"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
model = Sequential()
model.add(Conv2D(filters = 32,
                 kernel_size=(3, 3),
                 padding = 'same',
                 input_shape = (32, 32, 3), 
                 activation = 'relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters = 64,
                 kernel_size=(3, 3),
                 padding = 'same',
                 activation = 'relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation = 'softmax'))
print(model.summary())

"""[Trainning]"""
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
train_history = model.fit(x_img_train_normailze,
                          y_label_train_OneHot,
                          validation_split = 0.2,
                          epochs = 10,
                          batch_size = 128,
                          verbose = 1)
						  
"""[Graph]"""
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

"""[Evaluate]"""
scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot, verbose = 1)
scores[1]

"""[Predictoin]"""
prediction = model.predict_classes(x_img_test_normalize)
prediction[:10]
def plot_images_labels_prediction(images, labels, prediction, idx, num):
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
plot_images_labels_prediction(x_img_test, y_label_test, prediction, 0, 10)

"""[Prediction Probability]"""
Prediction_Probability = model.predict(x_img_test_normalize)
def show_Prediction_Probability(y, prediction, x_img, Prediction_Probability, i):			#label, prediction ans, predict image, Probability, 開始顯示data
	print('label:', label_dict[y[i][0]], 'predict', label_dict[prediction[i]])				#顯示 label, prediction ans
	plt.figure(figsize = (2, 2))															#顯示image大小並畫出
	plt.imshow(np.reshape(x_img_test[i], (32, 32, 3)))
	plt.show()
	for j in range(10):																		#用迴圈顯示預測機率
		print(label_dict[j] + 'Probability: %1.9f'%(Prediction_Probability[i][j]))
show_Prediction_Probability(y_label_test, prediction, x_img_test, Prediction_Probability, 0)
show_Prediction_Probability(y_label_test, prediction, x_img_test, Prediction_Probability, 3)

"""[Confusion matrix]"""
prediction.shape
y_label_test.shape
y_label_test.reshpae(-1)		#轉成one dimensional array
import pandas as pd
print(label_dict)
pd.crosstab(y_label_test.reshpae(-1), prediction, rownames=['label'], colnames=['predict'])

"""[Three times convolution]"""
from keras.utils import np_utils
from keras.datasets import cifar10
import numpy as np
np.random.seed(10)
(x_img_train, y_label_train), \
(x_img_test, y_label_test) = cifar10.load_data()
label_dict = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
model = Sequential()
model.add(Conv2D(filters = 32,
                 kernel_size=(3, 3),
                 padding = 'same',
                 input_shape = (32, 32, 3), 
                 activation = 'relu'))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 32,
                 kernel_size=(3, 3),
                 padding = 'same',
                 input_shape = (32, 32, 3), 
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#~~~
model.add(Conv2D(filters = 64,
                 kernel_size=(3, 3),
                 padding = 'same',
                 activation = 'relu'))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 64,
                 kernel_size=(3, 3),
                 padding = 'same',
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#~~~
model.add(Conv2D(filters = 128,
                 kernel_size=(3, 3),
                 padding = 'same',
                 activation = 'relu'))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 128,
                 kernel_size=(3, 3),
                 padding = 'same',
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(2500, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1500, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation = 'softmax'))
print(model.summary())

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
train_history = model.fit(x_img_train_normailze,
                          y_label_train_OneHot,
                          validation_split = 0.2,
                          epochs = 50,
                          batch_size = 300,
                          verbose = 1)
scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot, verbose = 1)
scores[1]

"""[Save model weight]"""
train_history = model.fit(x_img_train_normailze,
                          y_label_train_OneHot,
                          validation_split = 0.2,
                          epochs = 5,
                          batch_size = 128,
                          verbose = 1)
#Before start trainning
try:
	model.load_weights("SaveModel/cifarCnnModel.h5")	#第二次開始會讀取先前權重
	print("loading model success, keep training")		
except:
	print("loading model failed, training new one")		#第一次都會顯示這個因為沒存過權重
#After training
model.save_weights("SaveModel/cifarCnnModel.h5")
print("Saved model to disk")
