"""[Data 預先處理]"""
#import model
from keras.utils import np_utils
import numpy as np
np.random.seed(10)
#read mnist data
from keras.datasets import mnist
(x_train_image, y_train_label), \
(x_test_image, y_test_label) = mnist.load_data()
#將28*28影像轉成784一維陣列(float表示)
x_Train = x_train_image.reshape(60000,784).astype('float32')
x_Test = x_test_image.reshape(10000,784).astype('float32')
#Normailze 提高預測精準度與加速收斂
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255
#label encoding by One-hot
y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)

"""[Construct Model]"""
#import model
from keras.models import Sequential
from keras.layers import Dense
#Build Sequential model, input_layer & hidden_layer & output_layer by DNN(Dense Neural Network)
model = Sequential()
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation = 'relu'))	#hidden_layer neurons個數, input_layer neurons個數, normal distribution常態分佈的亂數set weight & bias, define激活函數relu
model.add(Dense(units = 10, kernel_initializer = 'normal', activation = 'softmax'))				#output_layer neurons個數, normal distribution常態分佈的亂數set weight & bias, define激活函數softmax
#model info
print(model.summary())	#Shape:neurons個數, Param:上一層neurons個數*本層的 + 本層neurons個數

"""[附註公式]"""
#activation functionz(模擬neurons * 傳送訊息的軸突 + 長度會是本層neurons個數)
hidden = relu(input*weight_1 + bias_1)
output = softmax(hidden*weight_2 + bias_2)

"""[Train]"""
#setting trainning model
model.compile(loss = 'categorical_crossentropy',		#loss function : crossentropy
              optimizer = 'adam',						#optimizer : adam 使trainning更快收斂
              metrics = ['accuracy'])					#evaluate model準確率使用accuracy
#Use model.fit for trainning, training process will store in train_hiatory
train_history = model.fit(x = x_Train_normalize,		#feature	
                          y = y_Train_OneHot,			#label
                          validation_split = 0.2,		#trainng data(80%) verification data(20%)
                          epochs = 10,					#trainng 10 times in time
                          batch_size = 200,				#every batch 200 data
                          verbose = 2)					#show trainning process
				#Epoch 10/10 : 2s - loss: 0.0316 - acc: 0.9920 - val_loss: 0.0807 - val_acc: 0.9757
#Trainning Porcess Graph
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

"""[Evaluate data accuracy]"""
#Use model.evaluate evaluate accuracy
scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print()
print('accuracy = ', scores[1])

"""[Predict data]"""
#對原始資料做預測，輸出前10筆預測資料
prediction = model.predict_classes(x_Test)
prediction
#上一章函式預測
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
plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx = 340)

"""[Confusion matrix]"""
#混淆矩陣用途
容易混淆的值可用此矩陣顯示，也稱error matrix

#Build Confusion matrix
import pandas as pd
pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict'])
df = pd.DataFrame({'label':y_test_label, 'predict':prediction})
df[:2]
df[(df.label == 5)&(df.predict == 3)]	#顯示實際值為5但預測是3
plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx = 340, num = 1) #第340筆資料

"""[Hidden layer increased to 1000]"""
#將model.add中256改成1000查看準確率變化，發現準確率上升但overfitting嚴重
#Solve overfitting by Dropout
from keras.layers import Dropout		#Dropout的做法是在訓練過程中隨機地忽略一些神經元，正向傳播過程中對於下游神經元的貢獻效果暫時消失了，反向傳播時該神經元也不會有任何權重的更新
model.add(Dropout(0.5))					#在加完input layer時加入，train & validate accuracy 差距變小代表成功
#Solve overfitting by two hidden layers + Dropout
model.add(Dense(units=1000, kernel_initializer='normal', activation = 'relu'))	#多加一層方法
model.add(Dropout(0.5))															#時間變長，train & validate accuracy 差距更近