import tensorflow as tf
from numpy.random import RandomState
batch_size = 8
#定义两个输入节点
x = tf.placeholder(tf.float32,shape=(None,2),name = 'x-input')
#回归问题中一般只有一个输出节点
y_ = tf.placeholder(tf.float32,shape = (None,1),name = 'y-input')
#定义一个单层神经网络前向传播的过程，简单的加权和
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)
#定义两个参数，当预测多了之后的成本损失和预测少了之后的利润损失
loss_more = 10
loss_less = 1
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*loss_more,(y_-y)*loss_less))
#loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

#生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
'''
设置回归的正确值为两个输入的和加上一个随机量，这是为了加入不可预测的噪音，
否则不同损失函数的意义不大，因为不同损失函数都会在能完全预测正确的时候最低。
一般噪音为均值为0的小量，所以这里的噪音设置为-0.05-0.05的随机数。
'''
Y = [[x1+x2+rdm.rand()/10.0-0.05]for (x1,x2) in X]
print(rdm.rand()/10.0-0.05)
#训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    steps = 5000
    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})


