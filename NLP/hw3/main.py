import numpy as np
import random
import codecs
import csv

csv.field_size_limit(500 * 1024 * 1024)


class LDA:
    def __init__(self, data_txt, topic_count):
        self.topic_count = topic_count
        self.topic_word_count = {}  # 每个topic有多少词
        self.topic_word_fre = {}  # 每个topic的词频表，字典的列表
        for i in range(self.topic_count):
            self.topic_word_fre[i] = self.topic_word_fre.get(i, {})
        self.doc_word_from_topic = []  # 每篇文章中的每个词来自哪个topic
        self.doc_word_count = []  # 每篇文章中有多少词
        self.doc_topic_fre = []  # 每篇文章topic词频
        for data in data_txt:
            topic = []
            docfre = {}
            for word in data:
                a = random.randint(0, self.topic_count - 1)  # 为每个单词赋予一个随机初始topic
                topic.append(a)
                if '\u4e00' <= word <= '\u9fa5':
                    self.topic_word_count[a] = self.topic_word_count.get(a, 0) + 1  # 统计每个topic总词数
                    docfre[a] = docfre.get(a, 0) + 1  # 统计每篇文章对应topic的词频
                    self.topic_word_fre[a][word] = self.topic_word_fre[a].get(word, 0) + 1
            self.doc_word_from_topic.append(topic)
            for i in range(self.topic_count):
                docfre[i] = docfre.get(i, 0)
            docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
            self.doc_topic_fre.append(docfre)
            self.doc_word_count.append(sum(docfre))  # 统计每篇文章的总词数
        self.topic_word_count = list(
            dict(sorted(self.topic_word_count.items(), key=lambda x: x[0], reverse=False)).values())
        self.doc_topic_fre = np.array(self.doc_topic_fre)  # 转为array方便后续计算
        self.topic_word_count = np.array(self.topic_word_count)  # 转为array方便后续计算
        self.doc_word_count = np.array(self.doc_word_count)  # 转为array方便后续计算
        self.doc_topic_pro = []  # 每个topic被选中的概率
        self.doc_topic_pro_new = []  # 记录每次迭代后每个topic被选中的概率
        self.cal_pro()

    def cal_pro(self):
        # 计算一篇文章中每个topic被选中的概率
        for i in range(len(data_txt)):
            doc = np.divide(self.doc_topic_fre[i], self.doc_word_count[i])
            self.doc_topic_pro.append(doc)

    def train(self):
        stop = 0  # 迭代停止标志
        loopcount = 1  # 迭代次数
        while stop == 0:
            i = 0
            for data in data_txt:
                top = self.doc_word_from_topic[i]
                for w in range(len(data)):
                    word = data[w]
                    pro = []
                    topfre = []
                    if '\u4e00' <= word <= '\u9fa5':
                        for j in range(self.topic_count):
                            topfre.append(self.topic_word_fre[j].get(word, 0))
                        pro = self.doc_topic_pro[
                                  i] * topfre / self.topic_word_count  # 计算每篇文章选中各个topic的概率乘以该词语在每个topic中出现的概率，得到该词出现的概率向量
                        m = np.argmax(pro)  # 认为该词是由上述概率之积最大的那个topic产生的
                        self.doc_topic_fre[i][top[w]] -= 1  # 更新每个文档有多少各个topic的词
                        self.doc_topic_fre[i][m] += 1
                        self.topic_word_count[top[w]] -= 1  # 更新每个topic的总词数
                        self.topic_word_count[m] += 1
                        self.topic_word_fre[top[w]][word] = self.topic_word_fre[top[w]].get(word, 0) - 1  # 统计每个topic总词数
                        self.topic_word_fre[m][word] = self.topic_word_fre[m].get(word, 0) + 1  # 统计每个topic总词数
                        top[w] = m
                self.doc_word_from_topic[i] = top
                i += 1
            if loopcount == 1:  # 计算新的每篇文章选中各个topic的概率
                for i in range(len(data_txt)):
                    doc = np.divide(self.doc_topic_fre[i], self.doc_word_count[i])
                    self.doc_topic_pro_new.append(doc)
                self.doc_topic_pro_new = np.array(self.doc_topic_pro_new)
            else:
                for i in range(len(data_txt)):
                    doc = np.divide(self.doc_topic_fre[i], self.doc_word_count[i])
                    self.doc_topic_pro_new[i] = doc
            if (self.doc_topic_pro_new == self.doc_topic_pro).all():  # 如果每篇文章选中各个topic的概率不再变化，则认为模型已经训练完毕
                stop = 1
            else:
                self.doc_topic_pro = self.doc_topic_pro_new.copy()
            loopcount += 1
            print(loopcount)
        print(self.doc_topic_pro_new)  # 输出最终训练的到的每篇文章选中各个topic的概率
        print('模型训练完毕！')

    def test(self, test_txt):
        doc_word_from_topic = []  # 每篇文章中的每个词来自哪个topic
        doc_word_count = []  # 每篇文章中有多少词
        doc_topic_fre = []  # 每篇文章topic词频
        i = 0
        for data in test_txt:
            topic = []
            docfre = {}
            for word in data:
                a = random.randint(0, self.topic_count - 1)  # 为每个单词赋予一个随机初始topic
                topic.append(a)
                if '\u4e00' <= word <= '\u9fa5':
                    docfre[a] = docfre.get(a, 0) + 1  # 统计每篇文章的词频
            doc_word_from_topic.append(topic)
            for i in range(self.topic_count):
                docfre[i] = docfre.get(i, 0)
            docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
            doc_topic_fre.append(docfre)
            doc_word_count.append(sum(docfre))  # 统计每篇文章的总词数
            i += 1

        doc_topic_fre = np.array(doc_topic_fre)
        doc_word_count = np.array(doc_word_count)

        doc_topic_pro = []  # 每个topic被选中的概率
        doc_topic_pro_new = []  # 记录每次迭代后每个topic被选中的新概率
        for i in range(len(test_txt)):
            doc = np.divide(doc_topic_fre[i], doc_word_count[i])
            doc_topic_pro.append(doc)
        doc_topic_pro = np.array(doc_topic_pro)

        stop = 0  # 迭代停止标志
        loopcount = 1  # 迭代次数
        while stop == 0:
            i = 0
            for data in test_txt:
                top = doc_word_from_topic[i]
                for w in range(len(data)):
                    word = data[w]
                    pro = []
                    topfre = []
                    if '\u4e00' <= word <= '\u9fa5':
                        for j in range(self.topic_count):
                            topfre.append(self.topic_word_fre[j].get(word, 0))
                            # exec('topfre.append(Topic_fre{}.get(word, 0))'.format(j))  # 读取该词语在每个topic中出现的频数

                        pro = doc_topic_pro[
                                  i] * topfre / self.topic_word_count  # 计算每篇文章选中各个topic的概率乘以该词语在每个topic中出现的概率，得到该词出现的概率向量
                        m = np.argmax(pro)  # 认为该词是由上述概率之积最大的那个topic产生的
                        doc_topic_fre[i][top[w]] -= 1  # 更新每个文档有多少各个topic的词
                        doc_topic_fre[i][m] += 1
                        top[w] = m
                doc_word_from_topic[i] = top
                i += 1

            if loopcount == 1:  # 计算新的每篇文章选中各个topic的概率
                for i in range(len(test_txt)):
                    doc = np.divide(doc_topic_fre[i], doc_word_count[i])
                    doc_topic_pro_new.append(doc)
                doc_topic_pro_new = np.array(doc_topic_pro_new)
            else:
                for i in range(len(test_txt)):
                    doc = np.divide(doc_topic_fre[i], doc_word_count[i])
                    doc_topic_pro_new[i] = doc

            if (doc_topic_pro_new == doc_topic_pro).all():  # 如果每篇文章选中各个topic的概率不再变化，则认为训练集已分类完毕
                stop = 1
            else:
                doc_topic_pro = doc_topic_pro_new.copy()
            loopcount += 1

        print(loopcount)
        print('测试集测试完毕！')
        result = []
        for k in range(len(test_txt)):
            pro = []
            for i in range(len(data_txt)):
                dis = 0
                for j in range(self.topic_count):
                    dis += (self.doc_topic_pro[i][j] - doc_topic_pro[k][j]) ** 2  # 计算欧式距离
                pro.append(dis)
            m = pro.index(min(pro))
            result.append(m)
        print(result)
        return result


if __name__ == "__main__":
    with codecs.open('word_tr.csv', encoding='utf-8-sig') as f:
        data_txt = []
        name_list = []
        for row in csv.DictReader(f, skipinitialspace=True):
            name_list.append(row['label'])
            data_txt.append(eval(row['data']))

    with codecs.open('word.csv', encoding='utf-8-sig') as f:
        test_txt = []
        label_list = []
        for row in csv.DictReader(f, skipinitialspace=True):
            test_txt.append(eval(row['data']))
            label_list.append(row['label'])
    lda = LDA(data_txt, 100)
    lda.train()
    result = lda.test(test_txt)
    tr = 0
    error = 0
    for i in range(len(result)):
        print("number:{:<3d} label:{:<20} result:{:<20}".format(i + 1, label_list[i], name_list[result[i]]))
        if label_list[i] == name_list[result[i]]:
            tr += 1
        else:
            error += 1
    print("分类正确数：{} 分类错误数：{}".format(tr, error))
