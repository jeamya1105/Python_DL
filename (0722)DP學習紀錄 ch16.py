"""[Matrix operation simulate Neural network operation]"""
#Y_output = activation(X_input * Weight + bias)
X = tf.Variable([[0.4, 0.2, 0.4]])
W = tf.Variable([[-0.5, -0.2],
                 [-0.3, 0.4],
                 [-0.5, 0.2]])
b = tf.Variable([[0.1, 0.2]])
XWb = tf.matmul(X, W) + b
Y1 = tf.nn.relu(tf.matmul(X,W) + b)				#relu feature : >0才輸出值, <0就等於0
Y2 = tf.nn.sigmoid(tf.matmul(X,W) + b)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(XWb))
    print(sess.run(Y1))
    print(sess.run(Y2))
	
(_XWb, _Y1, _Y2) = sess.run((XWb, Y1, Y2))		#可一次得到所有_XWb, _Y1, _Y2
#Random variable
W = tf.Variable(tf.random_normal([3, 2]))		#3*2 matrix = 3rows * 2columns = 3列 * 2行
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
#layer 函數, Create 2 layers neurals network
def layer(output_dim, input_dim, inputs, activation = None):	#輸出neurons數量, 輸入neurons數量, 輸入的matrix, 函數
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs
#3 layers neurals network
X = tf.placeholder("float", [None, 4])
h = layer(output_dim = 3, input_dim = 4, inputs = X, activation = tf.nn.relu)	#hidden layer
y = layer(output_dim = 2, input_dim = 3, inputs = h)							#output layer
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4, 0.2, 0.4, 0.5]])
    (LX, Lh, Ly) = sess.run((X, h, y), feed_dict = {X:X_array})
    print(LX)
    print(Lh)
    print(Ly)
"""[Layer Debug, 可顯示weight & bias]"""
def layer_d(output_dim, input_dim, inputs, activation = None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs, W, b

X = tf.placeholder("float", [None, 4])
h, W1, b1 = layer_d(output_dim = 3, input_dim = 4, inputs = X, activation = tf.nn.relu)	#hidden layer
y, W2, b2 = layer_d(output_dim = 2, input_dim = 3, inputs = h)							#output layer
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4, 0.2, 0.4, 0.5]])
    (LX, Lh, Ly, w_1, w_2, b_1, b_2) = sess.run((X, h, y, W1, W2, b1, b2), feed_dict = {X:X_array})
    print(LX)
    print(w_1, b_1)
    print(Lh)
    print(w_2, b_2)
    print(Ly)
