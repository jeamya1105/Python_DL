"""[Keras]"""
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)
(x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()
x_Train4D = x_Train.reshape(x_Train.shape[0], 28, 28, 1).astype('float32')
x_Test4D = x_Test.reshape(x_Test.shape[0], 28, 28, 1).astype('float32')
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
model = Sequential()
model.add(Conv2D(filters = 16,
                 kernel_size=(5, 5),
                 padding = 'same',
                 input_shape = (28, 28, 1),
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters = 36,
                 kernel_size=(5, 5),
                 padding = 'same',
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))
print(model.summary())

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
train_history = model.fit(x = x_Train4D_normalize,
                          y = y_TrainOneHot,
                          validation_split = 0.2,
                          epochs = 10,
                          batch_size = 300,
                          verbose = 2)
						  
"""[Tensorflow]"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1), name = 'W')
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape = shape), name = 'b')
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides =[1,2,2,1], padding = 'SAME')
	
with tf.name_scope('Input_Layer'):
    x = tf.placeholder("float", shape = [None, 784], name = "x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
with tf.name_scope('C1_Conv'):
    W1 = weight([5,5,1,16])
    b1 = bias([16])
    Conv1 = conv2d(x_image, W1) + b1
    C1_Conv = tf.nn.relu(Conv1)
with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)
with tf.name_scope('C2_Conv'):
    W2 = weight([5,5,16,36])
    b2 = bias([36])
    Conv2 = conv2d(C1_Pool, W2) + b2
    C2_Conv = tf.nn.relu(Conv2)
with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_2x2(C2_Conv)
with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C2_Pool, [-1, 1764])
with tf.name_scope('D_Hidden_Layer'):
    W3 = weight([1764, 128])
    b3 = bias([128])
    D_Hidden = tf.nn.relu(tf.matmul(D_Flat, W3)+b3)
    D_Hidden_Dropout = tf.nn.dropout(D_Hidden, keep_prob = 0.8)
with tf.name_scope('Output_Layer'):
    W4 = weight([128, 10])
    b4 = bias([10])
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden_Dropout, W4)+b4)
	
with tf.name_scope('optimizer'):
    y_label = tf.placeholder("float", [None, 10])
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_predict, labels = y_label))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_function)
with tf.name_scope('evaluate_model'):
    correct_prediction = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
trainEpochs = 30
batchSize = 100
totalBatchs = int(mnist.train.num_examples / batchSize)
loss_list = [];epoch_list = [];accuracy_list = []
from time import time
startTime = time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        sess.run(optimizer, feed_dict = {x:batch_x, y_label:batch_y})
    loss, acc = sess.run([loss_function, accuracy], feed_dict = {x:mnist.validation.images, y_label:mnist.validation.labels})
    epoch_list.append(epoch);
    loss_list.append(loss)
    accuracy_list.append(acc)
    print("Train Epoch : ", '%02d' % (epoch+1), "Loss = ", "{:.9f}".format(loss), "Accuracy = ", acc)
duration = time() - startTime
print("Train Finished takes : ", duration)