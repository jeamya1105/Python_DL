"""[Create computational graph]"""
import tensorflow as tf
ts_c = tf.constant(2, name = 'ts_c')	#�`�ƦW��name = 2
ts_c									#(tf.Tensor 'ts_c:0' shape=() dtype=int32) = (Tensorflow �i�q, 0 dimension tensor, datatype)
ts_x = tf.Variable(ts_c+5, name='ts_x')	#�ܼƦW��name = ts_c + 5
sess = tf.Session()						#session object
init = tf.global_variables_initializer()#initial all variable
sess.run(init)
#two way to ��� tensorflow variable				
print(sess.run(ts_c))
print(ts_c.eval(session = sess))
sess.close()
#with���i�ϥ�session�A�X�h��session�|�۰�����
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	print(sess.run(ts_c))
	print(sess.run(ts_x))
#����computational graph setting variable
width = tf.placeholder("int32")
height = tf.placeholder("int32")
area = tf.multiply(width, height)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(area, feed_dict = {width : 6, height : 8}))
#tensorflow �ƭȹB��
https://www.tensorflow.org/api_guides/python/math_ops

"""[TensorBoard]"""
tf.summary.merge_all()											#�N�n��ܦbTensorBoard��ƾ�X
train_writer = tf.summary.FileWriter('log/area', sess.graph)	#data write in log file
#�bcmd��activate tensorflow -> tensorboard --logdir = c:\pythonwork\log\area

"""[1,2 dimension tensor]"""
#�إ߱i�q
ts_X = tf.Variable([0.4, 0.2, 0.4])								#�@���ɤJlist�Y�i(�V�q)�A���T�Ӽƭ�
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X = sess.run(ts_X)
    print(X)						
W = tf.Variable([[-0.5, 0.2],									#�G���ɤJlist�Y�i(�x�})�A�C�@������ӼƭȦ����T�����
                    [-0.3, 0.4], 
                    [-0.5, 0.2]])
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    W_array = sess.run(W)
    print(W_array)
print(X.shape, W_array.shape)									#(3,) (3, 2) -> x�����(�@��) �C��y�Ӽƭ�(�G��)
#�i�q�B��
X = tf.Variable([[1., 1., 1.]])
W = tf.Variable([[-0.5, -0.2],
                 [-0.3, 0.4],
                 [-0.5, 0.2]])
Y = tf.Variable([[0.1, 0.2]])
XWY = tf.matmul(X, W) + Y										#matmul��matrix �ۭ�
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(XWY))
#input �ʺA��J��
import numpy as np
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))
X = tf.placeholder("float", [None,3])							#"float"��data type, None���Ĥ@���׶ǤJ���Ƥ���, 3���ĤG���׬�����ƶq��3�ӭ�
Y1 = tf.nn.relu(tf.matmul(X,W) + b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4, 0.2, 0.4]])
    (_b, _W, _X, _Y) = sess.run((b, W, X, Y1), feed_dict = {X:X_array})
    print(_b)
    print(_W)
    print(_X)
    print(_Y)
#layer ���
def layer(output_dim, input_dim, inputs, activation = None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs