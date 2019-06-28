import tensorflow as tf
from tensorflow.python.framework import graph_util

#声明两个变量并计算它们的和
v1 = tf.Variable(tf.constant(1.0,shape = [1],name = "v1"))
v2 = tf.Variable(tf.constant(2.0,shape = [1],name = "v2"))
result = v1 + v2
init_op = tf.global_variables_initializer()

#声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess,"model.ckpt")

#使用和保存模型代码中一样的方式来声明变量
v1 = tf.Variable(tf.constant(1.0,shape = [1],name = "v1"))
v2 = tf.Variable(tf.constant(2.0,shape = [1],name = "v2"))
result1 = v1 + v2
saver = tf.train.Saver()
with tf.Session() as sess:
    #加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
    saver.restore(sess,"model.ckpt")
    print(sess.run(result1))


#不重复定义图上的运算，直接加载已经持久化的图
saver = tf.train.import_meta_graph("model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess,"model.ckpt")
    #通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

#变量及其取值的直接保存
v1 = tf.Variable(tf.constant(1.0,shape = [1]),name = "v1")
v2 = tf.Variable(tf.constant(2.0,shape = [1]),name = "v2")
result1 = v1+v2
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    # 将图中的变量及其取值转化为常量，同时将图中不必要的节点去掉。
    # add 表示需要保存的节点名称
    output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['add'])
    with tf.gfile.GFile("combined_model.ph","wb") as f:
        f.write(output_graph_def.SerializeToString())


#直接计算定义的加法结果
import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename = "combined_model.pb"
    #读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    #将graph_def中保存的图加载到当前的图中
    result2= tf.import_graph_def(graph_def,return_elements=["add:0"])
    print(sess.run(result2))

