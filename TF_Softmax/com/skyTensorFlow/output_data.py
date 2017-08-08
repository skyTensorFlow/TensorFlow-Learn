# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 09:35:59 2017

@author: chenbin
"""

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
x = tf.placeholder('float',[None,784])
"""x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
（这里的None表示此张量的第一个维度可以是任何长度的。）"""
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
"""我们赋予tf.Variable不同的初值来创建不同的Variable：在这里，
我们都用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置。"""

y = tf.nn.softmax(tf.matmul(x,W) + b) #matmul 矩阵相乘

y_ = tf.placeholder("float", [None,10]) #记录真实值

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#计算交叉熵,tf.redece_sum是求和

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
"""我们要求TensorFlow用梯度下降算法（gradient descent algorithm）
以0.01的学习速率最小化交叉熵。
"""

#init = tf.initialize_all_variables() #初始化变量 此方法被启用已经。
init=tf.global_variables_initializer() #初始化变量
sess = tf.Session()
sess.run(init) #在一个Session里面启动我们的模型，并且初始化变量

#训练模型
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
""" 
该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，next_batch随机选
然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
"""

#评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
"""
tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的
其数据最大值所在的索引值。由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，
比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
而 tf.argmax(y_,1) 代表正确的标签，
我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)
"""
#取平均值得到准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
"""将x或者x.values转换为dtype
tensor a is [1.8, 2.2], dtype=tf.float
tf.cast(a, tf.int32) ==> [1, 2] # dtype=tf.int32
"""
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
