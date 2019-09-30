# -*- coding: utf-8 -*-
import tensorflow  as tf

from tensorflow.examples.tutorials.mnist import input_data

from mnist import y_

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None,784])
y = tf.placeholder("float", shape=[None,10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return  tf.Variable(initial)

def conv2d(x,w):
    return  tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")

def max_poll_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#第一层，卷积层
#初始化Q为【】的张量，表示卷积核大小为5*5,第一层网络的输入输出神经元个数分别为1，32
W_conv1 = weight_variable([5,5,1,32])
#初始化b为【32】，即输出大小
b_conv1 = bias_variable([32])

#把输入x（二维张量，shape为[batch,784])变为4d的x_images,x_images的shape应该是[bantch,28,28,28,1]
#-1表示自动推测这个维度的size
x_image = tf.reshape(x,[-1,28,28,1])

#把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling
#h_pool1的输出即为第一层网络输出，shape为[batch,14,14,1]
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_poll_2x2(h_conv1)

#第二层，卷积层
#卷积核大小依然是5*5，这层的输入和输出神经元数为32和64
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = weight_variable([64])

#h_pool2几位第二层网络输出，shape为[batch,7,7,1]
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_poll_2x2(h_conv2)

#第三层，全连接层
#这层是拥有1024个神经元的全连接层
#W的第一维size为7*7*64，7*7是h_pool2输出size，64是第2层输出神经元个数
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

#计算前需要把第二层的输出reshape成[batch,7*7*64]的张量
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#Dropout层
#为了减少过拟合，在输出层前加入dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#输出层
#最后，添加一个softmax层
#可以理解为另一个全连接层，只不过输出时使用softmax将网络输出值转换成了概率
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+ b_fc2)

#预测值和真实值之间的交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_predict = tf.equal(tf.arg_max(y_conv,1),tf.argmax(y_,1))

#计算正确预测项的比例，因为tf.equal返回的是布尔值
#使用tf.cast把布尔值转换为浮点数，然后用tf.reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))
sess.run(tf.initialize_all_variables())

#开始训练
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 10:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

print(f"test accuracy %g" % accuracy.eval(feed_dect={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))