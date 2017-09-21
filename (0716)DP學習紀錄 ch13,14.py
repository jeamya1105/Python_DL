"""[Download datasets]"""
import urllib.request
import os
import tarfile			#�����Y��
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath = "data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('download : ',result)
#extract file
if not os.path.exists("data/aclImdb"):							#�P�_�ؿ��O�_�s�b
    tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')		#�}�����Y��
    result = tfile.extractall('data/')							#�����Y
	
"""[Read Data]"""
from keras.preprocessing import sequence							#list�I���ɵu��
from keras.preprocessing.text import Tokenizer					#�إߦr��Token��

import re
def rm_tags(text):
	re_tag = re.compile(r'<[^>]+>')							#�ܼƬ�'<[^>]+>'
	return re_tag.sub('', text)									#text ���ŦX�ܼƪ��������Ŧr��

def read_files(filetype):
	path = "data/aclImdb/"										#save path
	file_list=[]
	
	positive_path = path + filetype + "/pos/"					#postive �����ɮץؿ� positive_path
	for f in os.listdir(positive_path):							#positive_path�ؿ��Uall data�[�Jlist
		file_list+=[positive_path+f]
		
	negative_path = path + filetype + "/neg/"					#negative �����ɮץؿ� negative_path
	for f in os.listdir(negative_path):							#negative_path�ؿ��Uall data�[�Jlist
		file_list+=[negative_path+f]
	
	print('read', filetype, 'files:', len(file_list))			#total�ɮ׼ƶq
	all_labels = ([1] * 12500 + [0] * 12500)					#all_labels = ����+�t��
	all_texts = []
	for fi in file_list:										#�}��Ū�r��A��join�s���Ҧ����e
		with open(fi, encoding = 'utf8') as file_input:
			all_texts += [rm_tags(" ".join(file_input.readlines()))]
	
	return all_labels, all_texts
y_train, train_text = read_files("train")						#y_train��ܵ���0��1, train_text��ܦr��
y_test, test_text = read_files("test")

"""[Crearte Token]"""
token = Tokenizer(num_words = 2000)								#Crearte 2000 words �r��
token.fit_on_texts(train_text)									#�Ҧ�training�v���A�X�{���ƪ��e2000�ӦC�J
print(token.document_count)
print(token.word_index)

"""[��r��Ʀr]"""
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
print(train_text[0])
print(x_train_seq[0])
#�ഫ����׬ۦP�A�I���ɵu
x_train = sequence.pad_sequences(x_train_seq, maxlen = 100)
x_test = sequence.pad_sequences(x_test_seq, maxlen = 100)
print(len(x_train_seq[0]))
print(x_train_seq[0])
print(x_train[0])

"""[Crearte Embedding layer]"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding

model = Sequential()
model.add(Embedding(output_dim = 32,							#�ন32���צV�q
                     input_dim = 2000,							#2000�r�夺�e
                     input_length = 100))						#every list ��100��
					 
"""[Crearte model]"""
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.35))
model.add(Dense(units = 1, activation = 'sigmoid'))
print(model.summary())

"""[Training]"""
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
train_history = model.fit(x_train,
                          y_train,
                          validation_split = 0.2,
                          epochs = 10,
                          batch_size = 100,
                          verbose = 1)

"""[Evaluate]"""
scores = model.evaluate(x_test, y_test, verbose = 1)
scores[1]

"""[Prediction]"""
predict = model.predict_classes(x_test)
predict[:10]
predict_classes = predict.reshape(-1)				#2���}�C��1���}�C
predict_classes[:10]
SentimentDict = {1:'P~', 0:'N~'}
def display(i):
    print(test_text[i])
    print('label : ', SentimentDict[y_test[i]], 
          'prediction : ', SentimentDict[predict_classes[i]])
input_text = '''
I have been waiting for this movie for a long time, but after watching the movie I was really disappointed. It's not like I don't like this Genre, but this movie, which is the remake of the 1991 Disney Classic, just fell flat. 
As everyone says that "art is subjective", and it's really true, which can be seen from the various 'Good' rating of the film across different platforms.
In the end, you can give it a try, but only if you like the casting of the movie, and Yes! the sets are amazing(the only good thing about the movie), but one thing which bothered me a lot is the singing, which is literally garbage(not even trying to be politically correct).
I really wanted this to be good movie, but it's a really forgettable movie.
'''
input_seq = token.texts_to_sequences([input_text])	#�ন�Ʀr
pad_input_seq = sequence.pad_sequences(input_seq, maxlen = 100)		#max 100�Ʀr
predict_result = model.predict_classes(pad_input_seq)
predict_result[0][0]
SentimentDict[predict_result[0][0]]

"""[�ξ� function]"""
def predict_review(input_text):
	input_seq = token.texts_to_sequences([input_text])
	pad_input_seq = sequence.pad_sequences(input_seq, maxlen = 100)
	predict_result = model.predict_classes(pad_input_seq)
	print(SentimentDict[predict_result[0][0]])
	
"""[RNN model]"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
model = Sequential()
model.add(Embedding(output_dim = 32,
                     input_dim = 3800,
                     input_length = 380))
model.add(Dropout(0.35))
model.add(SimpleRNN(units = 16))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.35))
model.add(Dense(units = 1, activation = 'sigmoid'))
print(model.summary())

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
train_history = model.fit(x_train,
                          y_train,
                          validation_split = 0.2,
                          epochs = 10,
                          batch_size = 100,
                          verbose = 1)

scores = model.evaluate(x_test y_test, verbose = 1)
scores[1]

"""[LSTM]"""
model.add(SimpleRNN(units = 16)) �令 model.add(LSTM(32))