"""[Import trainning datasets]"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print(mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples)		#train data, validation data, test data

"""[Tainning data]"""
print(mnist.train.images.shape, mnist.train.labels.shape)									#trainning data = images + labels
len(mnist.train.images[0])
mnist.train.images[0]																		#�i�ݨ�all data in [0] �w�Q normalize
import matplotlib.pyplot as plt																#��ܹϧΥ�
def plot_image(image):
    plt.imshow(image.reshape(28, 28), cmap = 'binary')
    plt.show()
import numpy as np																			#���labels�Ʀr��
np.argmax(mnist.train.labels[0])

def plot_images_labels_prediction(images, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(np.reshape(images[idx], (28, 28)), cmap = 'binary')						#Tensorflow data�쥻�O�@���Ʀr(784), �ন�G���v��(28*28)�~����ܹϧ�
        title = "label = " +str(np.argmax(labels[idx]))										#�쥻�N�Oone_hot���, ���ର�쥻�Ʀr�Φ�
        if len(prediction) > 0:
            title += ", predixt = " + str(prediction[idx])
        ax.set_title(title, fontsize = 10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
plot_images_labels_prediction(mnist.train.images, mnist.train.labels, [], 0)
batch_images_xs, batch_labels_ys = mnist.train.next_batch(batch_size = 100)					#batcch size = 100�C��Ū100����ƶixs & ys
