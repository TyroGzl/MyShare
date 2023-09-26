import jieba
import jieba.analyse
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 中文字体可修改
mpl.rcParams['axes.unicode_minus'] = False


class ChineseDataSet:
    def __init__(self, name):
        self.data = None
        self.name = name
        # 单个字
        self.word = []  # 单个字列表
        self.word_len = 0  # 单个字总字数
        # 词
        self.split_word = []  # 单个词列表
        self.split_word_len = 0  # 单个词总数
        with open("tools/cn_stopwords.txt", "r", encoding='utf-8') as f:
            self.stop_word = f.read().split('\n')
            f.close()

    def read_file(self, filename=""):
        # 如果未指定名称，则默认为类名
        if filename == "":
            filename = self.name
        target = "ChineseDataSet/" + filename + ".txt"
        with open(target, "r", encoding='gbk', errors='ignore') as f:
            self.data = f.read()
            self.data = self.data.replace(
                '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
            f.close()
        # 分词
        for words in jieba.cut(self.data):
            if (words not in self.stop_word) and (not words.isspace()):
                self.split_word.append(words)
                self.split_word_len += 1
        # 统计字数
        for word in self.data:
            if (word not in self.stop_word) and (not word.isspace()):
                # if not word.isspace():
                self.word.append(word)
                self.word_len += 1

    def write_file(self):
        # 将文件内容写入总文件
        target = "ChineseDataSet/data.txt"
        with open(target, "a") as f:
            f.write(self.data)
            f.close()

    def get_unigram_tf(self, word):
        # 得到单个词的词频表
        unigram_tf = {}
        for w in word:
            unigram_tf[w] = unigram_tf.get(w, 0) + 1
        return unigram_tf

    def get_bigram_tf(self, word):
        # 得到二元词的词频表
        bigram_tf = {}
        for i in range(len(word) - 1):
            bigram_tf[(word[i], word[i + 1])] = bigram_tf.get(
                (word[i], word[i + 1]), 0) + 1
        return bigram_tf

    def get_trigram_tf(self, word):
        # 得到三元词的词频表
        trigram_tf = {}
        for i in range(len(word) - 2):
            trigram_tf[(word[i], word[i + 1], word[i + 2])] = trigram_tf.get(
                (word[i], word[i + 1], word[i + 2]), 0) + 1
        return trigram_tf

    def calc_entropy_unigram(self, word, is_ci=0):
        # 计算一元模型的信息熵
        word_tf = self.get_unigram_tf(word)
        word_len = sum([item[1] for item in word_tf.items()])
        entropy = sum(
            [-(word[1] / word_len) * math.log(word[1] / word_len, 2) for word in
             word_tf.items()])
        if is_ci:
            print("<{}>基于词的一元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
        else:
            print("<{}>基于字的一元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
        return entropy

    def calc_entropy_bigram(self, word, is_ci=0):
        # 计算二元模型的信息熵
        # 计算二元模型总词频
        word_tf = self.get_bigram_tf(word)
        last_word_tf = self.get_unigram_tf(word)
        bigram_len = sum([item[1] for item in word_tf.items()])
        entropy = []
        for bigram in word_tf.items():
            p_xy = bigram[1] / bigram_len  # 联合概率p(xy)
            p_x_y = bigram[1] / last_word_tf[bigram[0][0]]  # 条件概率p(x|y)
            entropy.append(-p_xy * math.log(p_x_y, 2))
        entropy = sum(entropy)
        if is_ci:
            print("<{}>基于词的二元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
        else:
            print("<{}>基于字的二元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
        return entropy

    def calc_entropy_trigram(self, word, is_ci):
        # 计算三元模型的信息熵
        # 计算三元模型总词频
        word_tf = self.get_trigram_tf(word)
        last_word_tf = self.get_bigram_tf(word)
        trigram_len = sum([item[1] for item in word_tf.items()])
        entropy = []
        for trigram in word_tf.items():
            p_xy = trigram[1] / trigram_len  # 联合概率p(xy)
            p_x_y = trigram[1] / last_word_tf[(trigram[0][0], trigram[0][1])]  # 条件概率p(x|y)
            entropy.append(-p_xy * math.log(p_x_y, 2))
        entropy = sum(entropy)
        if is_ci:
            print("<{}>基于词的三元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
        else:
            print("<{}>基于字的三元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
        return entropy


def my_plot(X, Y1, Y2, Y3, num):
    # 标签位置
    x = range(0, len(X))
    # 柱状图宽度
    width = 0.2
    # 各柱状图位置
    x1_width = [i - width * 2 for i in x]
    x2_width = [i - width for i in x]
    x3_width = [i for i in x]
    # 设置图片大小、绘制柱状图
    plt.figure(figsize=(19.2, 10.8))
    plt.bar(x1_width, Y1, fc="r", width=width, label="一元模型")
    plt.bar(x2_width, Y2, fc="b", width=width, label="二元模型")
    plt.bar(x3_width, Y3, fc="g", width=width, label="三元模型")
    # 设置x轴
    plt.xticks(x, X, rotation=40, fontsize=10)
    plt.xlabel('数据库', fontsize=10)
    # 设置y轴
    plt.ylabel('信息熵', fontsize=10)
    plt.ylim(0, max(Y1) + 2)
    # 标题
    if (num == 1):
        plt.title("以字为单位的信息熵", fontsize=10)
    elif num == 2:
        plt.title("以词为单位的信息熵", fontsize=10)
    # 标注柱状图上方文字
    autolabel(x1_width, Y1)
    autolabel(x2_width, Y2)
    autolabel(x3_width, Y3)

    plt.legend()
    plt.savefig('chinese' + str(num) + '.png')
    plt.show()


def autolabel(x, y):
    for a, b in zip(x, y):
        plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=10)


if __name__ == "__main__":
    data_set_list = []
    # 每次运行程序将总内容文件清空
    with open("ChineseDataSet/data.txt", "w") as f:
        f.close()
    with open("log.txt", "w") as f:
        f.close()
    # 读取小说名字
    with open("ChineseDataSet/inf.txt", "r") as f:
        txt_list = f.read().split(',')
        i = 0
        for name in txt_list:
            locals()[f'set{i}'] = ChineseDataSet(name)
            data_set_list.append(locals()[f'set{i}'])
            i += 1
        f.close()
    # 分别针对每本小说进行操作
    word_unigram_entropy, word_bigram_entropy, word_trigram_entropy, words_unigram_entropy, words_bigram_entropy, words_trigram_entropy = [], [], [], [], [], []
    for set in data_set_list:
        set.read_file()
        set.write_file()
        # 字为单位
        word_unigram_entropy.append(set.calc_entropy_unigram(set.word, 0))
        word_bigram_entropy.append(set.calc_entropy_bigram(set.word, 0))
        word_trigram_entropy.append(set.calc_entropy_trigram(set.word, 0))
        # 词为单位
        words_unigram_entropy.append(set.calc_entropy_unigram(set.split_word, 1))
        words_bigram_entropy.append(set.calc_entropy_bigram(set.split_word, 1))
        words_trigram_entropy.append(set.calc_entropy_trigram(set.split_word, 1))
        with open("log.txt", "a") as f:
            f.write("{:<10} 字数：{:10d} 词数：{:10d} 信息熵：{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}\n".format(set.name,
                                                                                                               set.word_len,
                                                                                                               set.split_word_len,
                                                                                                               word_unigram_entropy[
                                                                                                                   -1],
                                                                                                               word_bigram_entropy[
                                                                                                                   -1],
                                                                                                               word_trigram_entropy[
                                                                                                                   -1],
                                                                                                               words_unigram_entropy[
                                                                                                                   -1],
                                                                                                               words_bigram_entropy[
                                                                                                                   -1],
                                                                                                               words_trigram_entropy[
                                                                                                                   -1]))
            f.close()
    # 对所有小说进行操作
    set_total = ChineseDataSet("total")
    set_total.read_file("data")
    word_unigram_entropy.append(set_total.calc_entropy_unigram(set_total.word, 0))
    word_bigram_entropy.append(set_total.calc_entropy_bigram(set_total.word, 0))
    word_trigram_entropy.append(set_total.calc_entropy_trigram(set_total.word, 0))

    words_unigram_entropy.append(set_total.calc_entropy_unigram(set_total.split_word, 1))
    words_bigram_entropy.append(set_total.calc_entropy_bigram(set_total.split_word, 1))
    words_trigram_entropy.append(set_total.calc_entropy_trigram(set_total.split_word, 1))

    with open("log.txt", "a") as f:
        f.write(
            "{:<10} 字数：{:10d} 词数：{:10d} 信息熵：{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}、{:.4f}\n".format(set_total.name,
                                                                                                       set_total.word_len,
                                                                                                       set_total.split_word_len,
                                                                                                       word_unigram_entropy[
                                                                                                           -1],
                                                                                                       word_bigram_entropy[
                                                                                                           -1],
                                                                                                       word_trigram_entropy[
                                                                                                           -1],
                                                                                                       words_unigram_entropy[
                                                                                                           -1],
                                                                                                       words_bigram_entropy[
                                                                                                           -1],
                                                                                                       words_trigram_entropy[
                                                                                                           -1]))
        f.close()
    # 绘图
    x_label = [i.name for i in data_set_list]
    x_label.append(set_total.name)

    my_plot(x_label, word_unigram_entropy, word_bigram_entropy, word_trigram_entropy, 1)
    my_plot(x_label, words_unigram_entropy, words_bigram_entropy, words_trigram_entropy, 2)
