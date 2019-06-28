import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import numpy as np
from matplotlib import pyplot as plt

BATCH_SIZE = 100  #一个训练batch中的训练数据个数
LEARNING_RATE_BASE = 0.8  #基础的学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率

REGULARIZATION_RATE = 0.0001  #描述模型复杂度的正则项在损失函数中的系数
TRAINING_STEPS = 5000  #训练轮数
MOVING_AVERAGE_DECAY = 0.99  #滑动平均衰减率

#模型保存的路径名
model_save_path = "F:\PythonProject\MachineLearning\MLProject\\NER\\NER-master\\test\CNN\model_save"
model_name ="model.ckpt"


def plot(valid,loss):
    fig = plt.figure(figsize=(10,6),dpi = 80)
    print(len(valid))
    x = np.linspace(0,len(valid),len(valid))
    plt.plot(x,valid,label = "valid_value")
    plt.plot(x,loss,label = "loss_value")
    plt.xlabel("Epoch")
    plt.ylabel("valid_acc/loss_value")
    plt.title("CNN-train acc&&loss")
    plt.legend()
    plt.savefig("mnist_CNNfig.png")
    plt.show()



def train(mnist):
    # 在网络节点输入的数据，根据具体的训练数据决定
    x = tf.placeholder(tf.float32, [BATCH_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.IMAGE_SIZE,
                                    mnist_inference.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None,mnist_inference.NUM_LABELS], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x,True,regularizer)
    global_step = tf.Variable(0,trainable =False)
    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    # 参数优化
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 在训练神经网络模型时，每过一遍数据急需要通过反向传播更新网络参数，又要更新每一个参数的滑动平均值。
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    #初始化持久化类
    saver = tf.train.Saver()
    accValue = []
    lossValue = []
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            xs = np.reshape(xs,(BATCH_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS))
            _,loss_value,step,acc = sess.run([train_op,loss,global_step,accuracy],feed_dict={x:xs,y_:ys})
            accValue.append(acc)
            lossValue.append(loss_value)
            # 每1000轮输出以此在验证集上的测试结果
            if i % 1000 == 0:
                '''
                输出当前训练情况
                '''
                print("After %d training step(s),loss on training batch is %g" % (step, loss_value))
                saver.save(sess,os.path.join(model_save_path,model_name),global_step = global_step)
    plot(accValue,lossValue)
def main(argv = None):
    mnist = input_data.read_data_sets("T:\BaiduNetdiskDownload\Dataset\MnistDataset\\",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
