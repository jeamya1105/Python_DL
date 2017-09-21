"""[Create computational graph]"""
import tensorflow as tf
ts_c = tf.constant(2, name = 'ts_c')	#常數名稱name = 2
ts_c									#(tf.Tensor 'ts_c:0' shape=() dtype=int32) = (Tensorflow 張量, 0 dimension tensor, datatype)
ts_x = tf.Variable(ts_c+5, name='ts_x')	#變數名稱name = ts_c + 5
sess = tf.Session()						#session object
init = tf.global_variables_initializer()#initial all variable
sess.run(init)
#two way to 顯示 tensorflow variable				
print(sess.run(ts_c))
print(ts_c.eval(session = sess))
sess.close()
#with內可使用session，出去後session會自動關閉
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	print(sess.run(ts_c))
	print(sess.run(ts_x))
#執行computational graph setting variable
width = tf.placeholder("int32")
height = tf.placeholder("int32")
area = tf.multiply(width, height)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(area, feed_dict = {width : 6, height : 8}))
#tensorflow 數值運算
https://www.tensorflow.org/api_guides/python/math_ops

"""[TensorBoard]"""
tf.summary.merge_all()											#將要顯示在TensorBoard資料整合
train_writer = tf.summary.FileWriter('log/area', sess.graph)	#data write in log file
#在cmd打activate tensorflow -> tensorboard --logdir = c:\pythonwork\log\area

"""[1,2 dimension tensor]"""
#建立張量
ts_X = tf.Variable([0.4, 0.2, 0.4])								#一維導入list即可(向量)，有三個數值
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X = sess.run(ts_X)
    print(X)						
W = tf.Variable([[-0.5, 0.2],									#二維導入list即可(矩陣)，每一筆有兩個數值但有三筆資料
                    [-0.3, 0.4], 
                    [-0.5, 0.2]])
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    W_array = sess.run(W)
    print(W_array)
print(X.shape, W_array.shape)									#(3,) (3, 2) -> x筆資料(一維) 每筆y個數值(二維)
#張量運算
X = tf.Variable([[1., 1., 1.]])
W = tf.Variable([[-0.5, -0.2],
                 [-0.3, 0.4],
                 [-0.5, 0.2]])
Y = tf.Variable([[0.1, 0.2]])
XWY = tf.matmul(X, W) + Y										#matmul為matrix 相乘
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(XWY))
#input 動態輸入值
import numpy as np
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))
X = tf.placeholder("float", [None,3])							#"float"為data type, None為第一維度傳入筆數不限, 3為第二維度美筆資料量為3個值
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
#layer 函數
def layer(output_dim, input_dim, inputs, activation = None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs