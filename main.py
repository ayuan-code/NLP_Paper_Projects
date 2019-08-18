from get_all_data import Dataset
from model import model
import tensorflow as tf
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = str(1) #设置gpu num，没有gpu随便设置会自动使用cpu
data = Dataset()

session_config = tf.ConfigProto(
                    log_device_placement=False,
                    inter_op_parallelism_threads=0,
                    intra_op_parallelism_threads=0,
                    allow_soft_placement=True)
session_config.gpu_options.allow_growth = True # 使tensorflow能顾动态申请显存，而不是一下占满
m = model(train_first=True, train_second=False)
with tf.Session(config=session_config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # 将这句话注释去掉可以续跑，因为训练句向量训练很多轮，50以上，并且非常慢，所以如果觉得训练的不够多，可以继续训练，而不是从头训练
    # saver.restore(sess, "./model/result.ckpt")
    for i in range(50):
        m.train(sess,data.train_word_datas,data.train_para_datas,data.train_new_labels,512)
        saver.save(sess,"./model/result.ckpt")
tf.reset_default_graph() #每一次都清空计算图并重新创建。
m = model(train_first=False, train_second=True)
with tf.Session(config=session_config) as sess:
    saver = tf.train.Saver()
    saver.restore(sess,"./model/result.ckpt")
    train_para = np.reshape(np.array(range(25000)),[25000,1])
    for i in range(100):
        m.train_mlp(sess,train_para[0:20000],data.train_labels[0:20000],32) # 训练
        m.test_mlp(sess, train_para[20000:], data.train_labels[20000:], 32) # 验证
    saver.save(sess,"./model/result.ckpt")
tf.reset_default_graph()
m = model(train_first=False, train_second=False)
with tf.Session(config=session_config) as sess:
    saver = tf.train.Saver()
    saver.restore(sess,"./model/result.ckpt")
    for i in range(50):
        m.train(sess, data.test_word_datas, data.test_para_datas, data.test_new_labels, 512,is_train=False)
    saver.save(sess,"./model/result.ckpt")
tf.reset_default_graph()
m = model(train_first=False, train_second=False)
with tf.Session(config=session_config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./model/result.ckpt")
    test_para = np.reshape(np.array(range(25000)),[25000,1])
    m.test_mlp(sess, test_para, data.test_labels, 32)
    saver.save(sess, "./model/result.ckpt")