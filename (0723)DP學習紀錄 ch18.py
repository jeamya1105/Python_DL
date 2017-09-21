"""[Data prepare]"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

"""[Create model]"""
#layer function
def layer(output_dim, input_dim, inputs, activation = None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))		#weight隨機取
    b = tf.Variable(tf.random_normal([1, output_dim]))				#bias隨機取
    XWb = tf.matmul(inputs, W) + b									#input * weight + bias
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs
#input layer, 有多筆資料進入, 每筆資料有784個值
x = tf.placeholder("float", [None, 784])
#hidden layer, (hidden layer neurons, input layer neurons, input value, function)
h1 = layer(output_dim = 256, input_dim = 784, inputs = x, activation = tf.nn.relu)
#output layer, (output layer neurons, hidden layer neurons, hidden layer value, function)
y_predict = layer(output_dim = 10, input_dim = 256, inputs = h1, activation = None)

y_label = tf.placeholder("float", [None, 10])
#cross entropy算平均
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_predict, labels = y_label))
#內建optimizer, 依照誤差值更新weight & bias
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_function)
#每筆資料預測equal or not
correct_prediction = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_predict, 1))
#tf.cast 將 correct_prediction轉成float用 tf.reduce_mean算平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

"""[Trainning]"""
trainEpochs = 15
batchSize = 100
totalBatchs = int(mnist.train.num_examples / batchSize)
#initial 誤差, 訓練週期, 準確率
loss_list = [];epoch_list = [];accuracy_list = []
from time import time
startTime = time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#15 times Epoch training
for epoch in range(trainEpochs):
	#55000/100 times training
    for i in range(totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batchSize)					#x = feature , y = label
        sess.run(optimizer, feed_dict = {x:batch_x, y_label:batch_y})
	#此次training loss
    loss, acc = sess.run([loss_function, accuracy], feed_dict = {x:mnist.validation.images, y_label:mnist.validation.labels})
	#存入list並顯示結果
    epoch_list.append(epoch);
    loss_list.append(loss)
    accuracy_list.append(acc)
    print("Train Epoch : ", '%02d' % (epoch+1), "Loss = ", "{:.9f}".format(loss), "Accuracy = ", acc)
duration = time() - startTime
print("Train Finished takes : ", duration)

"""[Graph]"""
%matplotlib inline									#顯示在jupyter視窗
import matplotlib.pyplot as plt
fig = plt.gcf()										#get figure graph
fig.set_size_inches(4, 2)							#graph size
plt.plot(epoch_list, loss_list, label = 'loss')		#(xlabel data, ylabel data, 線的名稱)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc = 'upper left')			#圖例顯示在左上角, 名稱loss

plt.plot(epoch_list, accuracy_list, label = 'accuracy')
fig = plt.gcf()
fig.set_size_inches(4, 2)
plt.ylim(0.8, 1)									#ylabel顯示範圍
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

"""[Evaluate accuracy]"""
sess.run(accuracy, feed_dict = {x:mnist.test.images, y_label:mnist.test.labels})
#give test datasets 進行 prediction
pre_result = sess.run(tf.argmax(y_predict, 1), feed_dict = {x:mnist.test.images})
import numpy as np
def plot_images_labels_prediction(images, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(np.reshape(images[idx], (28, 28)), cmap = 'binary')
        title = "label = " +str(np.argmax(labels[idx]))
        if len(prediction) > 0:
            title += ", predixt = " + str(prediction[idx])
        ax.set_title(title, fontsize = 10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
plot_images_labels_prediction(mnist.test.images, mnist.test.labels, pre_result, 0)

"""[Increase accuracy]"""
1. increase hidden layer neurons
2. increase hidden later_2

