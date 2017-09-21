"""[Data prepare]"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

"""[Create model]"""
#layer function
def layer(output_dim, input_dim, inputs, activation = None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))		#weight�H����
    b = tf.Variable(tf.random_normal([1, output_dim]))				#bias�H����
    XWb = tf.matmul(inputs, W) + b									#input * weight + bias
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs
#input layer, ���h����ƶi�J, �C����Ʀ�784�ӭ�
x = tf.placeholder("float", [None, 784])
#hidden layer, (hidden layer neurons, input layer neurons, input value, function)
h1 = layer(output_dim = 256, input_dim = 784, inputs = x, activation = tf.nn.relu)
#output layer, (output layer neurons, hidden layer neurons, hidden layer value, function)
y_predict = layer(output_dim = 10, input_dim = 256, inputs = h1, activation = None)

y_label = tf.placeholder("float", [None, 10])
#cross entropy�⥭��
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_predict, labels = y_label))
#����optimizer, �̷ӻ~�t�ȧ�sweight & bias
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_function)
#�C����ƹw��equal or not
correct_prediction = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_predict, 1))
#tf.cast �N correct_prediction�নfloat�� tf.reduce_mean�⥭��
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

"""[Trainning]"""
trainEpochs = 15
batchSize = 100
totalBatchs = int(mnist.train.num_examples / batchSize)
#initial �~�t, �V�m�g��, �ǽT�v
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
	#����training loss
    loss, acc = sess.run([loss_function, accuracy], feed_dict = {x:mnist.validation.images, y_label:mnist.validation.labels})
	#�s�Jlist����ܵ��G
    epoch_list.append(epoch);
    loss_list.append(loss)
    accuracy_list.append(acc)
    print("Train Epoch : ", '%02d' % (epoch+1), "Loss = ", "{:.9f}".format(loss), "Accuracy = ", acc)
duration = time() - startTime
print("Train Finished takes : ", duration)

"""[Graph]"""
%matplotlib inline									#��ܦbjupyter����
import matplotlib.pyplot as plt
fig = plt.gcf()										#get figure graph
fig.set_size_inches(4, 2)							#graph size
plt.plot(epoch_list, loss_list, label = 'loss')		#(xlabel data, ylabel data, �u���W��)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc = 'upper left')			#�Ϩ���ܦb���W��, �W��loss

plt.plot(epoch_list, accuracy_list, label = 'accuracy')
fig = plt.gcf()
fig.set_size_inches(4, 2)
plt.ylim(0.8, 1)									#ylabel��ܽd��
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

"""[Evaluate accuracy]"""
sess.run(accuracy, feed_dict = {x:mnist.test.images, y_label:mnist.test.labels})
#give test datasets �i�� prediction
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

