"""[Data �w���B�z]"""
#import model
from keras.utils import np_utils
import numpy as np
np.random.seed(10)
#read mnist data
from keras.datasets import mnist
(x_train_image, y_train_label), \
(x_test_image, y_test_label) = mnist.load_data()
#�N28*28�v���ন784�@���}�C(float���)
x_Train = x_train_image.reshape(60000,784).astype('float32')
x_Test = x_test_image.reshape(10000,784).astype('float32')
#Normailze �����w����ǫ׻P�[�t����
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
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation = 'relu'))	#hidden_layer neurons�Ӽ�, input_layer neurons�Ӽ�, normal distribution�`�A���G���ü�set weight & bias, define�E�����relu
model.add(Dense(units = 10, kernel_initializer = 'normal', activation = 'softmax'))				#output_layer neurons�Ӽ�, normal distribution�`�A���G���ü�set weight & bias, define�E�����softmax
#model info
print(model.summary())	#Shape:neurons�Ӽ�, Param:�W�@�hneurons�Ӽ�*���h�� + ���hneurons�Ӽ�

"""[��������]"""
#activation functionz(����neurons * �ǰe�T�����b�� + ���׷|�O���hneurons�Ӽ�)
hidden = relu(input*weight_1 + bias_1)
output = softmax(hidden*weight_2 + bias_2)

"""[Train]"""
#setting trainning model
model.compile(loss = 'categorical_crossentropy',		#loss function : crossentropy
              optimizer = 'adam',						#optimizer : adam ��trainning��֦���
              metrics = ['accuracy'])					#evaluate model�ǽT�v�ϥ�accuracy
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
#���l��ư��w���A��X�e10���w�����
prediction = model.predict_classes(x_Test)
prediction
#�W�@���禡�w��
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
#�V�c�x�}�γ~
�e���V�c���ȥi�Φ��x�}��ܡA�]��error matrix

#Build Confusion matrix
import pandas as pd
pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict'])
df = pd.DataFrame({'label':y_test_label, 'predict':prediction})
df[:2]
df[(df.label == 5)&(df.predict == 3)]	#��ܹ�ڭȬ�5���w���O3
plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx = 340, num = 1) #��340�����

"""[Hidden layer increased to 1000]"""
#�Nmodel.add��256�令1000�d�ݷǽT�v�ܤơA�o�{�ǽT�v�W�ɦ�overfitting�Y��
#Solve overfitting by Dropout
from keras.layers import Dropout		#Dropout�����k�O�b�V�m�L�{���H���a�����@�ǯ��g���A���V�Ǽ��L�{�����U�寫�g�����^�m�ĪG�Ȯɮ����F�A�ϦV�Ǽ��ɸӯ��g���]���|�������v������s
model.add(Dropout(0.5))					#�b�[��input layer�ɥ[�J�Atrain & validate accuracy �t�Z�ܤp�N���\
#Solve overfitting by two hidden layers + Dropout
model.add(Dense(units=1000, kernel_initializer='normal', activation = 'relu'))	#�h�[�@�h��k
model.add(Dropout(0.5))															#�ɶ��ܪ��Atrain & validate accuracy �t�Z���