import os
import nltk
import numpy as np
import random
from collections import Counter
import pickle
class Dataset():

    def __init__(self):
        if not os.path.exists("./data/train_word_datas"):
            train_datas, train_labels, train_unsup, test_datas, test_labels = self.get_all_datas()
            word2id = self.get_all_words(train_datas,train_unsup)
            train_datas = self.convert_data_word_to_id(word2id,train_datas)
            train_unsup = self.convert_data_word_to_id(word2id,train_unsup)
            test_datas = self.convert_data_word_to_id(word2id,test_datas)
            self.train_datas = train_datas
            self.train_labels = train_labels
            self.train_unsup = train_unsup
            self.test_datas = test_datas
            self.test_labels = test_labels
            # 这里可以只是self.train_datas，也可以是self.train_datas+self.train_unsup
            new_word_datas,new_papr_datas,new_labels = self.convert_data_to_new_data(self.train_datas)
            self.train_word_datas = new_word_datas
            self.train_para_datas = new_papr_datas
            self.train_new_labels = new_labels
            new_word_datas, new_papr_datas, new_labels = self.convert_data_to_new_data(self.test_datas)
            self.test_word_datas = new_word_datas
            self.test_para_datas = new_papr_datas
            self.test_new_labels = new_labels
            pickle.dump(self.train_word_datas,open("./data/train_word_datas","wb"))
            pickle.dump(self.train_para_datas,open("./data/train_para_datas","wb"))
            pickle.dump(self.train_new_labels,open("./data/train_new_labels","wb"))
            pickle.dump(self.train_labels,open("./data/train_labels","wb"))
            pickle.dump(self.test_word_datas,open("./data/test_word_datas","wb"))
            pickle.dump(self.test_para_datas,open("./data/test_para_datas","wb"))
            pickle.dump(self.test_new_labels,open("./data/test_new_labels","wb"))
            pickle.dump(self.test_labels,open("./data/test_labels","wb"))
        else:
            self.train_word_datas = pickle.load(open("./data/train_word_datas","rb"))
            self.train_para_datas = pickle.load(open("./data/train_para_datas","rb"))
            self.train_para_datas = self.train_para_datas.reshape([self.train_para_datas.shape[0],1])
            self.train_new_labels = pickle.load(open("./data/train_new_labels","rb"))
            self.train_labels = pickle.load(open("./data/train_labels","rb"))
            self.test_word_datas = pickle.load(open("./data/test_word_datas","rb"))
            self.test_para_datas = pickle.load(open("./data/test_para_datas","rb"))
            self.test_para_datas = self.test_para_datas.reshape([self.test_para_datas.shape[0],1])
            self.test_new_labels = pickle.load(open("./data/test_new_labels","rb"))
            self.test_labels = pickle.load(open("./data/test_labels","rb"))



    def get_data(self,path):
        '''
        根据文件的路径读取文件路径下的所有文件
        :param path: 文件路径
        :return: 所有的文本数据
        nltk如果出现错误，可以先装nltk，然后在python中输入nltk.download("popular")
        '''
        datas = []
        paths = os.listdir(path)
        paths = [path +file_name for file_name in paths]
        for i,file in enumerate(paths):
            if i%1000==0:
                print (i,len(paths))
            data = open(file,"r").read()
            data = data.lower()
            data = nltk.word_tokenize(data)
            datas.append(data)
        return datas
    def get_all_datas(self):
        '''
        得到所有的训练句子，无监督句子和测试句子。
        :return: 返回训练句子，训练标签，无监督句子，测试句子，测试标签
        '''
        train_neg_datas = self.get_data(path="/home/liuhui/python_files/paper_1/data/aclImdb/train/neg/")
        train_pos_datas = self.get_data(path="/home/liuhui/python_files/paper_1/data/aclImdb/train/pos/")
        train_unsup = self.get_data(path="/home/liuhui/python_files/paper_1/data/aclImdb/train/unsup/")
        test_neg_datas = self.get_data(path = "/home/liuhui/python_files/paper_1/data/aclImdb/test/neg/")
        test_pos_datas = self.get_data(path="/home/liuhui/python_files/paper_1/data/aclImdb/test/pos/")
        train_datas = train_neg_datas+train_pos_datas
        train_labels = [0]*len(train_neg_datas)+[1]*len(train_pos_datas)
        test_datas = test_neg_datas+train_pos_datas
        test_labels = [0]*len(test_neg_datas)+[1]*len(test_pos_datas)
        tmp = list(zip(train_datas,train_labels))
        random.shuffle(tmp)
        train_datas[:],train_labels[:] = zip(*tmp)
        tmp = list(zip(test_datas,test_labels))
        random.shuffle(tmp)
        test_datas[:],test_labels[:] = zip(*tmp)
        print (len(train_datas),len(train_labels))
        print (len(train_unsup))
        print (len(test_datas),len(test_labels))
        return train_datas,train_labels,train_unsup,test_datas,test_labels
    def get_all_words(self,train_datas,train_unsup):
        '''
        从训练句子和无监督句子中统计所有出现过的词以及它们的频率并取出现频率最高的29998个词加上pad和unk构建一个30000大小的词表
        :param train_datas: 所有的训练句子
        :param train_unsup: 所有的无监督句子
        :return: 30000大小的词典，每个词对应一个id
        '''
        all_words = []
        for sentence in train_datas:
            all_words.extend(sentence)
        for sentence in train_unsup:
            all_words.extend(sentence)
        count = Counter(all_words)
        count = dict(count.most_common(29998))
        word2id = {"<pad>":0,"<unk>":1}
        for word in count:
            word2id[word] = len(word2id)
        return word2id
    def convert_data_word_to_id(self,word2id,datas):
        '''
        将datas里面的词都转化正对应的id
        :param word2id: 30000大小的词典
        :param datas: 需要转化的数据
        :return: 返回转化完的数据
        '''
        for i,sentence in enumerate(datas):
            for j,word in enumerate(sentence):
                datas[i][j] = word2id.get(word,1)
        return datas
    def convert_data_to_new_data(self,datas):
        '''
        根据句子生成窗口大小为10的语言模型训练集，当句子长度不够10时需要在前面补pad。
        :param datas: 句子，可以只使用训练句子，也可以使用训练句子+无监督句子，后续需要训练更久。
        :return: 返回窗口大小为10的训练集，句子id和词标签。
        '''
        new_word_datas = []
        new_papr_datas = []
        new_labels = []
        for i,data in enumerate(datas):
            if i%1000==0:
                print (i,len(datas))
            for j in range(len(data)):
                if len(data)<10: # 如果句子长度不够10，开始pad
                    tmp_words = [0]*(10-len(data))+data[0:-1]
                    if set(tmp_words)=={1}: #同样，连续9个词都是unk就舍去
                        break
                    new_word_datas.append(tmp_words)
                    new_papr_datas.append(i)
                    new_labels.append(data[-1])
                    break
                tmp_words = data[j:j+9]
                if set(tmp_words)=={1}: # 开始发现存在连续出现unk的句子，这种句子没有意义，所以连续9个词都是unk，那么就舍去
                    continue
                new_papr_datas.append(i)
                new_word_datas.append(tmp_words)
                new_labels.append(data[j+9])
                if j+9+1==len(data): # 到最后10个单词break
                    break
        new_word_datas = np.array(new_word_datas)
        new_papr_datas = np.array(new_papr_datas)
        new_labels = np.array(new_labels)
        print (new_word_datas.shape)
        print (new_papr_datas.shape)
        print (new_labels.shape)
        return new_word_datas,new_papr_datas,new_labels


if __name__=="__main__":
    data = Dataset()

