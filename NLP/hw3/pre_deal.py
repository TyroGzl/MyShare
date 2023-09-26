import jieba
import jieba.analyse
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import csv


def read_file(filename, is_word=False):
    # 如果未指定名称，则默认为类名
    target = "data/" + filename + ".txt"
    with open(target, "r", encoding='gbk', errors='ignore') as f:
        data = f.read()
        data = data.replace(
            '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        f.close()
    # 分词
    with open("tools/cn_stopwords.txt", "r", encoding='utf-8') as fp:
        stop_word = fp.read().split('\n')
        fp.close()
    split_word = []
    if is_word:
        for word in data:
            if (word not in stop_word) and (not word.isspace()):
                split_word.append(word)
    else:
        for words in jieba.cut(data):
            if (words not in stop_word) and (not words.isspace()):
                split_word.append(words)
    return split_word


def save_train(is_word):
    with open("data/inf.txt", "r") as f:
        txt_list = f.read().split(',')
        dict_list = []
        for name in txt_list:
            data = read_file(name, is_word)
            dict = {
                'name': name,
                'data': data
            }
            dict_list.append(dict)
    if is_word:
        save_path = 'word_train.csv'
    else:
        save_path = 'words_train.csv'
    with open(save_path, 'a', newline='', encoding='utf-8') as fp:
        csv_header = ['name', 'data']  # 设置表头，即列名
        csv_writer = csv.DictWriter(fp, csv_header)
        if fp.tell() == 0:
            csv_writer.writeheader()
        csv_writer.writerows(dict_list)  # 写入数据



def extract_paragraph(is_word):
    # 读取小说名字
    word_len = 0
    with open("data/inf.txt", "r") as f:
        txt_list = f.read().split(',')
        dict = {}
        for name in txt_list:
            data = read_file(name, is_word)
            word_len += len(data)
            dict[name] = data
        f.close()
    # 计算每篇文章抽取段落数
    number = 1
    con_list = []
    for name in txt_list:
        count = int(len(dict[name]) / word_len * 200 + 0.5)
        # 特殊处理
        if name == '越女剑':
            count = 1
        pos = int(len(dict[name]) // count)
        for i in range(count):
            data_temp = dict[name][i * pos:i * pos + 500]
            con = {
                'number': number,
                'label': name,
                'data': data_temp
            }
            con_list.append(con)
            number += 1
    if is_word:
        save_path = 'word.csv'
    else:
        save_path = 'words.csv'
    with open(save_path, 'a', newline='', encoding='utf-8') as fp:
        csv_header = ['number', 'label', 'data']  # 设置表头，即列名
        csv_writer = csv.DictWriter(fp, csv_header)
        if fp.tell() == 0:
            csv_writer.writeheader()
        csv_writer.writerows(con_list)  # 写入数据

def extract_paragraph1(is_word):
    # 读取小说名字
    word_len = 0
    with open("data/inf.txt", "r") as f:
        txt_list = f.read().split(',')
        dict = {}
        for name in txt_list:
            data = read_file(name, is_word)
            word_len += len(data)
            dict[name] = data
        f.close()
    # 计算每篇文章抽取段落数
    number = 1
    con_list = []
    for name in txt_list:
        count = int(len(dict[name]) / word_len * 200 + 0.5)
        # 特殊处理
        if name == '越女剑':
            count = 1
        pos = int(len(dict[name]) // count)
        data_temp=[]
        for i in range(count):
            data_temp = data_temp + dict[name][i * pos:i * pos + 500]
        con = {
            'number': number,
            'label': name,
            'data': data_temp
        }
        con_list.append(con)
        number += 1
    if is_word:
        save_path = 'word_tr.csv'
    else:
        save_path = 'words_tr.csv'
    with open(save_path, 'a', newline='', encoding='utf-8') as fp:
        csv_header = ['number', 'label', 'data']  # 设置表头，即列名
        csv_writer = csv.DictWriter(fp, csv_header)
        if fp.tell() == 0:
            csv_writer.writeheader()
        csv_writer.writerows(con_list)  # 写入数据



if __name__ == "__main__":
    # extract_paragraph(True)
    # extract_paragraph(False)
    # save_train(True)
    # save_train(False)
    extract_paragraph1(True)
    extract_paragraph1(False)
