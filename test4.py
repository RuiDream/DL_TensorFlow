import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np
# Mnist数据集相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
#神经网络的参数
LAYER1_NODE = 500  # 隐藏层结点数
BATCH_SIZE = 100  #一个训练batch中的训练数据个数
LEARNING_RATE_BASE = 0.8  #基础的学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率

REGULARIZATION_RATE = 0.0001  #描述模型复杂度的正则项在损失函数中的系数
TRAINING_STEPS = 30000  #训练轮数
MOVING_AVERAGE_DECAY = 0.99  #滑动平均衰减率


def plot(valid,loss):
    fig = plt.figure(figsize=(10,6),dpi = 80)
    print(len(valid))
    x = np.linspace(0,len(valid)*100,len(valid))
    plt.plot(x,valid,label = "valid_value")
    plt.plot(x,loss,label = "loss_value")
    plt.xlabel("Epoch")
    plt.ylabel("valid_acc/loss_value")
    plt.legend()
    plt.savefig("mnist_fig.png")
    plt.show()


def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    #当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        #计算隐藏层的前向传播结果，这里使用了ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:
        #首先使用avg_class.average函数计算得出变量的滑动平均值
        #然后再计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

def train(mnist):
    #在网络节点输入的数据，根据具体的训练数据决定
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name = 'x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = 'y-input')
    #隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape = [LAYER1_NODE]))
    #输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev = 0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))
    # 计算输出值
    y = inference(x,None,weights1,biases1,weights2,biases2)
    global_step = tf.Variable(0,trainable=False)
    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)
    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = tf.argmax(y_,1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    regularization = regularizer(weights1)+regularizer(weights2)
    loss = cross_entropy_mean + regularization
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples,LEARNING_RATE_DECAY)
    #参数优化
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
    #在训练神经网络模型时，每过一遍数据急需要通过反向传播更新网络参数，又要更新每一个参数的滑动平均值。
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #定义损失值和验证集上的准确率
    lossValue = []
    validValue = []
    #初始化会话并开始训练过程
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        # 初始化变量
        sess.run(init_op)
        #划分验证数据，一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练结果
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        #准备测试数据，作为模型优劣的最后评价标准
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}
        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            #每1000轮输出以此在验证集上的测试结果
            if i%100 == 0:
                '''
                为了计算方便，本样例程序没有将验证数据划分为更小的batch。
                当神经网络模型比较复杂或者验证数据比较大时，太大的batch会导致计算时间过长甚至发生内存溢出的错误
                '''
                validate_acc,loss_value = sess.run([accuracy,loss],feed_dict=validate_feed)
                lossValue.append(loss_value)
                validValue.append(validate_acc)
                print("i:",str(i))
                #print("After %d training step(s),validation accuracy using average model is %g"%(i,validate_acc))
                # 产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        #在训练结束之后，在测试数据集上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training step(s),test accuracy using average model is %g" %(TRAINING_STEPS,test_acc))
    print("vaildValue:",str(validValue))
    print("lossValue:", str(lossValue))
    #画出验证集准确率和损失值
    plot(validValue,lossValue)


def main(argv =None):
    #声明处理MNIST数据集的类，这个类在初始化时自动下载数据
    mnist = input_data.read_data_sets("T:\BaiduNetdiskDownload\Dataset\MnistDataset\\",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    #提供了一个函数入口，默认执行函数main，也可以设置tf.app.run(main=test)
    tf.app.run()


