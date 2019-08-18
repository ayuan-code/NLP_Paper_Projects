import tensorflow as tf
from sklearn.metrics import accuracy_score
class model():
    def __init__(self,train_first,train_second):
        self.window = 10 #使用连续的9个词预测下一个词
        self.para_num = 75000
        self.create_placeholder()
        self.model(train_first,train_second)
    def create_placeholder(self):
        '''
        创建图的输入placeholder
        self.word_input: n-gram前n-1个词的输入
        self.para_input：篇章id的输入
        self.word_label: 语言模型预测下一个词的词标签
        self.label：这句话属于正类还是负类的类别标签
        :return:
        '''
        self.word_input = tf.placeholder(dtype=tf.int32,shape=[None,self.window-1])
        self.para_input = tf.placeholder(dtype=tf.int32,shape=[None,1])
        self.word_label = tf.placeholder(dtype=tf.int32,shape=[None])
        self.label = tf.placeholder(dtype=tf.int32,shape=[None])
    def model(self,train_first,train_second):
        '''
        :param train_first: 当train_first为True时，表示训练训练集的词向量和句向量
        :param train_second:  当train_second为True时，表示固定词向量和句向量，开始训练单隐层神经网络分类器用于情感分类
        当train_first和train_second都是False的时候表示测试阶段
        :return:
        '''
        with tf.variable_scope("train_parameters"):
            self.train_para_embedding = tf.Variable(initial_value=tf.truncated_normal(shape=[self.para_num,400]),
                                                    trainable=True,name="train_para_embedding")
            self.word_embedding = tf.Variable(initial_value=tf.truncated_normal(shape=[30000,400]),
                                              trainable=True,name="word_embedding")
        with tf.variable_scope("test_parameters"):
            self.test_para_embedding = tf.Variable(initial_value=tf.truncated_normal(shape=[25000,400]),trainable=True,
                                                   name="test_para_embedding")
        if train_first or train_second:
            para_input = tf.nn.embedding_lookup(self.train_para_embedding, self.para_input)  # batch_size*1*400
        else:
            para_input = tf.nn.embedding_lookup(self.test_para_embedding,self.para_input)
        word_inut = tf.nn.embedding_lookup(self.word_embedding,self.word_input)              #batch_size*9*400

        input = tf.concat([word_inut,para_input],axis=1) #batch_size*10*400
        input = tf.layers.flatten(input) #batch_size*4000
        with tf.variable_scope("train_parameters"):
            output = tf.layers.dense(input,units=30000,name="word_output")
        labels = tf.one_hot(self.word_label,30000)
        train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"train_parameters")
        test_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"test_parameters")
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-10), tf.trainable_variables())
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=output))+reg

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss_op,var_list=train_var)
        self.test_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss_op,var_list=test_var)

        mlp_input = tf.reshape(para_input,[-1,400])
        with tf.variable_scope("classification_parameters"):
            h1 = tf.layers.dense(mlp_input,units=50,activation="relu",trainable=True,name="h1")
            mlp_output = tf.layers.dense(h1,2,trainable=True,name="mlp_output")
        mlp_labels = tf.one_hot(self.label,2)
        self.mlp_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=mlp_labels,logits=mlp_output))
        classification_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classification_parameters")
        self.mlp_train_op = tf.train.AdamOptimizer(learning_rate=0.02).minimize(self.mlp_loss_op,var_list=classification_var)
        self.predict_op = tf.argmax(mlp_output,axis=1)
    def train(self,sess,word_datas,para_datas,word_label,batch_size,is_train=True):
        '''
        :param sess: tensorflow的Session，用来运行计算图
        :param word_datas: 所有的训练集word词组，大小为m*9，m为样本个数
        :param para_datas: 所有的训练集段落id组，大小为m
        :param word_label: 所有的词标签，大小为m
        :param batch_size: batch_size,是一个标量
        :param is_train: 训练的时候和测试的时候都使用这个函数，所以这是一个标志位，标注是训练还是测试
        :return: 无
        '''
        index = 0
        while index<len(word_datas):
            word_data_batch = word_datas[index:index+batch_size]
            para_data_batch = para_datas[index:index+batch_size]
            word_label_batch = word_label[index:index+batch_size]
            if is_train:
                loss,_ = sess.run([self.loss_op,self.train_op],feed_dict={self.word_input:word_data_batch,self.para_input:para_data_batch,
                                                                        self.word_label:word_label_batch})
            else:
                loss, _ = sess.run([self.loss_op, self.test_op],
                                   feed_dict={self.word_input: word_data_batch, self.para_input: para_data_batch,
                                              self.word_label: word_label_batch})
            if index%(batch_size*100)==0:
                print ("Train loss is:",loss)
                print (index,len(word_datas))
                if loss<1:
                    print (word_data_batch)
                    print (word_label_batch)
            index+=batch_size
    def train_mlp(self,sess,para_datas,labels,batch_size):
        '''
        :param sess: tensorflow的Session
        :param para_datas: 所有训练句子id，大小为25000
        :param labels: 所有句子的情感标签，大小为25000
        :param batch_size: 标量
        :return: 无
        '''
        index = 0
        while index < len(para_datas):
            para_data_batch = para_datas[index:index + batch_size]
            label_batch = labels[index:index+batch_size]
            loss,_ = sess.run([self.mlp_loss_op,self.mlp_train_op],feed_dict={self.para_input:para_data_batch,
                                                                              self.label:label_batch})
            if index%(batch_size*100)==0:
                #print ("Train loss is:",loss)
                #print (index,len(para_datas))
                pass
            index+=batch_size
    def test_mlp(self,sess,para_datas,labels,batch_size):
        '''
        :param sess: tensorflow的Session
        :param para_datas: 所有的测试句子id，大小为25000维的向量
        :param labels:  所有的测试句子标签，大小为25000维的向量，用来测试模型结果
        :param batch_size:  标量。
        :return: 无
        '''
        index=0
        result = []
        while index < len(para_datas):
            para_data_batch = para_datas[index:index + batch_size]
            pred = sess.run(self.predict_op,feed_dict={self.para_input:para_data_batch})
            result+=list(pred)
            index+=batch_size

        acc = accuracy_score(y_true=labels,y_pred=result)
        print ("Test acc is:",acc)



