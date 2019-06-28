import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
#加载mnist_inference.py 和mnist_train.py中定义的常量和函数
import  mnist_inference
import mnist_train
import numpy as np
#每一秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAl_INTERVAL_SEC = 10
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        #定义输入输出的格式
        x = tf.placeholder(tf.float32, [None, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,
                                        mnist_inference.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name = 'y-input')
        validate_feed = {x:tf.reshape(mnist.validation.images,[5000, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,
                                        mnist_inference.NUM_CHANNELS]),y_:mnist.validation.labels}
        y = mnist_inference.inference(x,False,None)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        #每隔EVAL_INTERVAL_SEC秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                #tf.train.get_checkpoint_state函数通过checkpoing文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    #加载模型
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('/')[-1]
                    accuracy_score,result = sess.run([accuracy,y],feed_dict=validate_feed)
                    print("After %s training step(s),validation accuracy = %g"%(global_step,accuracy_score))
                    print("result=",sess.run(tf.argmax(result,1)))
                else:
                    print('No checkpoing file found')
                    return
def main(argv = None):
    mnist = input_data.read_data_sets("T:\BaiduNetdiskDownload\Dataset\MnistDataset\\",one_hot=True)
    evaluate(mnist)

if __name__ =='__main__':
    tf.app.run()
