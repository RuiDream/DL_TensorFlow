
# 计算机存储单个像素点所用到的bit位称之为图像的深度.
# 图像的通道是指颜色三元素分别的像素值，若为RGB则图像的通道为3.

import tensorflow as tf
import numpy as np
# Mnist数据集相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积层的尺寸和深度
conv1_deep = 32
conv1_size = 5
#第二层卷积层的尺寸和深度
conv2_deep = 64
conv2_size = 5
#全连接层的节点个数
fc_size = 512



'''
定义卷积神经网络的前向传播过程，这里添加了一个新的参数train，用于区分训练过程和测试过程。
使用dropout方法，用于提升模型可靠性并防止过拟合，只用于训练过程
'''
def inference(input_tensor,train,regularizer):
    #第一层，输出28*28*32的矩阵
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[conv1_size,conv1_size,NUM_CHANNELS,conv1_deep],
        initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias",[conv1_deep],initializer=tf.constant_initializer(0.0))
        #使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding = 'SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    #第二层，最大池化，输入为28*28*32，输出为14*14*32
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize = [1,2,2,1],strides = [1,2,2,1],padding='SAME')
    #第三层，输入为14*14*32，输出为14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight",[conv2_size,conv2_size,conv1_deep,conv2_deep],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias",[conv2_deep],initializer=tf.constant_initializer(0.0))
        #使用边长为5，深度为64的过滤器，移动步长为1，使用全0填充
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides = [1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    #第四层，最大池化，输入为14*14*64，输出为7*7*64的矩阵
    with tf.name_scope('layer3-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #第五层，将7*7*64的矩阵拉为一个向量。
    #每一层神经网络的输入输出都为一个batch矩阵，所以这里得到的维度包含了一个batch中数据的个数
    pool_shape = pool2.get_shape().as_list()
    #计算将矩阵拉直成向量之后的长度，这个长度是矩阵长款及深度的成绩。
    #pool_shape[0]是一个batch中数据的个数
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    #通过tf.reshape函数将输出变成一个batch向量，一个向量长度为3136
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])
    #第五层输入为3136，输出为512。
    #引入dropout,在训练时会随机将部分节点的输出改为0，避免出现过拟合问题。
    #dropout一般只在全连接层而不是卷积层或者池化层使用
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight",[nodes,fc_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #将全连接层的权重加入正则化
        if regularizer !=None:
            tf.add_to_collection("losses",regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias",[fc_size],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)
    #第六层，输入为512的向量，输出为长度为10的向量，通过softmax得到最后的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight",[fc_size,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection("losses",regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias",[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,fc2_weights)+fc2_biases
    #返回第六层的输出
    return logit