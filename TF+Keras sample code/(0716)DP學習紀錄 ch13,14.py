"""[Download datasets]"""
import urllib.request
import os
import tarfile			#解壓縮用
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath = "data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('download : ',result)
#extract file
if not os.path.exists("data/aclImdb"):							#判斷目錄是否存在
    tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')		#開啟壓縮檔
    result = tfile.extractall('data/')							#解壓縮
	
"""[Read Data]"""
from keras.preprocessing import sequence							#list截長補短用
from keras.preprocessing.text import Tokenizer					#建立字典Token用

import re
def rm_tags(text):
	re_tag = re.compile(r'<[^>]+>')							#變數為'<[^>]+>'
	return re_tag.sub('', text)									#text 中符合變數的替換成空字串

def read_files(filetype):
	path = "data/aclImdb/"										#save path
	file_list=[]
	
	positive_path = path + filetype + "/pos/"					#postive 評價檔案目錄 positive_path
	for f in os.listdir(positive_path):							#positive_path目錄下all data加入list
		file_list+=[positive_path+f]
		
	negative_path = path + filetype + "/neg/"					#negative 評價檔案目錄 negative_path
	for f in os.listdir(negative_path):							#negative_path目錄下all data加入list
		file_list+=[negative_path+f]
	
	print('read', filetype, 'files:', len(file_list))			#total檔案數量
	all_labels = ([1] * 12500 + [0] * 12500)					#all_labels = 正面+負面
	all_texts = []
	for fi in file_list:										#開檔讀字串，用join連接所有內容
		with open(fi, encoding = 'utf8') as file_input:
			all_texts += [rm_tags(" ".join(file_input.readlines()))]
	
	return all_labels, all_texts
y_train, train_text = read_files("train")						#y_train表示評價0或1, train_text表示字串
y_test, test_text = read_files("test")

"""[Crearte Token]"""
token = Tokenizer(num_words = 2000)								#Crearte 2000 words 字典
token.fit_on_texts(train_text)									#所有training影評，出現次數的前2000個列入
print(token.document_count)
print(token.word_index)

"""[文字轉數字]"""
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
print(train_text[0])
print(x_train_seq[0])
#轉換後長度相同，截長補短
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
model.add(Embedding(output_dim = 32,							#轉成32維度向量
                     input_dim = 2000,							#2000字典內容
                     input_length = 100))						#every list 有100個
					 
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
predict_classes = predict.reshape(-1)				#2維陣列轉1維陣列
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
input_seq = token.texts_to_sequences([input_text])	#轉成數字
pad_input_seq = sequence.pad_sequences(input_seq, maxlen = 100)		#max 100數字
predict_result = model.predict_classes(pad_input_seq)
predict_result[0][0]
SentimentDict[predict_result[0][0]]

"""[統整 function]"""
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
model.add(SimpleRNN(units = 16)) 改成 model.add(LSTM(32))