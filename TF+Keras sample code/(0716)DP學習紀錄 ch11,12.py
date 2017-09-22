"""[Download datasets]"""
import urllib.request
import os				#os model make sure file enable or not
#Download
url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath = "data/titanic3.xls"							#save path, �n�ؤ@�Ӹ�Ƨ�data
if not os.path.isfile(filepath):						#if file doesn't exist then download
	result = urllib.request.urlretrieve(url, filepath)
	print('download : ',result)

"""[Data �w���B�z]"""
import numpy
import pandas as pd
all_df = pd.read_excel(filepath)
all_df[:2]							#survival �� label
#choose the frame we need to dataframe
cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
all_df = all_df[cols]
all_df[:2]
#name �b train���ݭn�ҥH�i����
df = all_df.drop(['name'], axis = 1)
#��Xnull�Ȩç令������
age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)
fare_mean = df['fare'].mean()
df['fare'] = df['fare'].fillna(fare_mean)
#sex���ȧ令0 1���
df['sex'] = df['sex'].map({'female':0, 'male':1}).astype(int)
#Embarked ���T�ӭȥ�OneHot�ഫ
x_OneHot_df = pd.get_dummies(data = df, columns = ["embarked" ])
x_OneHot_df[:2]
#dataframe to array
ndarray = x_OneHot_df.values
ndarray.shape
#get label & feature
Label = ndarray[:, 0]				#python slice�y�k [�Ҧ���Ƽ�, ��0���(label)]
Features = ndarray[:, 1:]			#python slice�y�k [�Ҧ���Ƽ�, ��1�̫ܳ����(feature)]
Label[:2]
Features[:2]
#feature normalize
from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler(feature_range = (0, 1))		#normalize range between 0 & 1
scaledFeatures = minmax_scale.fit_transform(Features)					#Feature �i�� normalize
scaledFeatures[:2]
#train & test data split
msk = numpy.random.rand(len(all_df)) < 0.8								#�H���H8:2��
train_df = all_df[msk]
test_df = all_df[~msk]
print(len(all_df), len(train_df), len(test_df))
#�ξ㦨�@��Data�w���B�z�禡
from sklearn import preprocessing
def PreprocessData(raw_df):
	df = raw_df.drop(['name'], axis = 1)
	age_mean = df['age'].mean()
	df['age'] = df['age'].fillna(age_mean)
	fare_mean = df['fare'].mean()
	df['fare'] = df['fare'].fillna(fare_mean)
	df['sex'] = de['sex'].map({'female':0, 'male':1}).astype(int)
	x_OneHot_df = pd.get_dummies(data = df, columns = ["embarked"])
	
	ndarray = x_OneHot_df.values
	Label = ndarray[:, 0]
	Features = ndarray[:, 1:]
	minmax_scale = preprocessing.MinMaxScaler(feature_range = (0, 1))
	scaledFeatures = minmax_scale.fit_transform(Features)
	
	return scaledFeatures, Label
#Train & Test data �w���B�z
train_Features, train_Label = PreprocessData(train_df)
test_Features, test_Label = PreprocessData(test_df)
train_Features[:2]
train_Label[:2]

"""[Build model]"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(units = 40,
                input_dim = 9,
                kernel_initializer = 'uniform',
                activation = 'relu'))
model.add(Dense(units = 30,
                kernel_initializer = 'uniform',
                activation = 'relu'))
model.add(Dense(units = 1,
                kernel_initializer = 'uniform',
                activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
train_history = model.fit(train_Features,
                          train_Label,
                          validation_split = 0.1,
                          epochs = 30,
                          batch_size = 30,
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

"""[Create forecast data]"""
#Series ��J���
Jack = pd.Series([0, 'Jack', 3, 'male', 23, 1, 0, 5.0000, 'S'])
Rose = pd.Series([1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S'])
#�إߨ�HDataFrame
JR_df = pd.DataFrame([list(Jack), list(Rose)], 
                     columns = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])
#�[�J�Ҧ�DataFrame�|�b�̫���
all_df = pd.concat([all_df, JR_df])
#�s�W��ӤH�A�ҥHfeature & label�n���s��
all_Features, Label = PreprocessData(all_df)
#�w���A�i�o��C�ӭ��Ȧs�����v
all_probability = model.predict(all_Features)
#�s�����v�[�JDataFrame
pd = all_df
pd.insert(len(all_df.columns),
          'survive probability', all_probability)
		  
"""[Find]"""
pd[(pd['survived'] == 0) & (pd['survive probability'] > 0.9)]
pd[:5]