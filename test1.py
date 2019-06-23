import tensorflow as tf
from numpy.random import RandomState
import os
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def plot(xRange,lossValue):
    fig = plt.figure(figsize=(20,8),dpi = 80)
    x = range(xRange)
    plt.plot(x,lossValue)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("fig.png")
    plt.show()

#完整神经网络解决框架
if __name__ == '__main__':
    #训练数据batch的大小
    batch_size = 8
    #定义神经网络参数
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
    #在shape的一个维度上使用None可以方便使用不大的batch大小。
    x = tf.placeholder(tf.float32,shape=(None,2),name = 'x-input')
    y_ = tf.placeholder(tf.float32,shape=(None,1),name = 'y-input')
    #定义神经网络前向传播的过程
    a = tf.matmul(x,w1)
    y = tf.matmul(a,w2)
    #定义损失函数和反向传播的算法
    #tf.clip_by_value()是指将数据y限制在一定范围之内，防止越界没有意义。
    #当y小于1e-10时取1e-10，当大于1时取1
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
    train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)
    #通过随机数生成一个模拟数据集
    ##1为随机种子，只要随机种子seed相同，产生的随机数序列就相同
    rdm = RandomState(1)
    dataset_size = 128
    #生成128*2的随机数
    X = rdm.rand(dataset_size,2)
    Y = [[int(x1+x2<1)] for (x1,x2) in X]
    loss_value =[]

    #创建一个会话来运行TensorFlow程序
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        #初始化变量
        sess.run(init_op)
        print(sess.run(w1))
        print(sess.run(w2))
        steps = 3000  #训练的轮数
        for i in range(steps):
            #每次选取batch_size个样本进行训练
            start = (i*batch_size)%dataset_size
            end = min(start+batch_size,dataset_size)
            #通过选取的样本训练神经网络并更新参数
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            loss_value.append(total_cross_entropy)
            if i % 500 == 0:
                print("After %d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy))
        print(sess.run(w1))
        print(sess.run(w2))
    plot(steps,loss_value)