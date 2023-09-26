# coding: utf-8
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import os
import jieba
import jieba.analyse


class lstm_model(nn.Module):
    def __init__(self, vocab, hidden_size, num_layers, dropout=0.5):
        super(lstm_model, self).__init__()
        self.vocab = vocab  # 字符数据集
        # 索引-字符互相转换字典
        self.int2char = {i: char for i, char in enumerate(vocab)}
        self.char2int = {char: i for i, char in self.int2char.items()}
        # 对字符进行one-hot encoding
        self.encoder = OneHotEncoder(sparse=True).fit(vocab.reshape(-1, 1))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # lstm层
        self.lstm = nn.LSTM(len(vocab), hidden_size, num_layers, batch_first=True, dropout=dropout)
        # 全连接层
        self.linear = nn.Linear(hidden_size, len(vocab))

    def forward(self, sequence, hs=None):
        out, hs = self.lstm(sequence, hs)  # lstm的输出格式（batch_size, sequence_length, hidden_size）
        out = out.reshape(-1, self.hidden_size)  # 这里需要将out转换为linear的输入格式，即（batch_size * sequence_length, hidden_size）
        output = self.linear(out)
        return output, hs

    def onehot_encode(self, data):
        return self.encoder.transform(data)

    def onehot_decode(self, data):
        return self.encoder.inverse_transform(data)

    def label_encode(self, data):
        return np.array([self.char2int[ch] for ch in data])

    def label_decode(self, data):
        return np.array([self.int2char[ch] for ch in data])


def get_batches(data, batch_size, seq_len):
    num_features = data.shape[1]
    num_chars = batch_size * seq_len  # 一个batch_size的长度
    num_batches = int(np.floor(data.shape[0] / num_chars))  # 计算出有多少个batches
    need_chars = num_batches * num_chars  # 计算出需要的总字符量
    targets = np.vstack((data[1:].A, data[0].A))  # 可能版本问题，取成numpy比较好reshape
    inputs = data[:need_chars].A.astype("int")  # 从原始数据data中截取所需的字符数量need_words
    targets = targets[:need_chars]
    targets = targets.reshape(batch_size, -1, num_features)
    inputs = inputs.reshape(batch_size, -1, num_features)
    for i in range(0, inputs.shape[1], seq_len):
        x = inputs[:, i: i + seq_len]
        y = targets[:, i: i + seq_len]
        yield x, y  # 节省内存


def train(model, data, batch_size, seq_len, epochs, lr=0.01):
    dir_name = os.path.basename(__file__).split(".")[0]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("The device used is: {}".format(device))
    print("Load model or not: \n1: Yes\n2: No\n")
    flag = eval(input())
    if flag == 1:
        print("Input model name: ")
        model_name = input()
        model.load_state_dict(torch.load(model_name))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    data = model.onehot_encode(data.reshape(-1, 1))

    train_loss = []

    for epoch in range(epochs):
        model.train()
        hs = None
        train_ls = 0.0
        for x, y in get_batches(data, batch_size, seq_len):
            optimizer.zero_grad()
            x = torch.tensor(x).float().to(device)
            out, hs = model(x, hs)
            hs = ([h.data for h in hs])
            y = y.reshape(-1, len(model.vocab))
            y = model.onehot_decode(y)
            y = model.label_encode(y.squeeze())
            y = torch.from_numpy(y).long().to(device)
            loss = criterion(out, y.squeeze())
            loss.backward()
            optimizer.step()
            train_ls += loss.item()

        train_loss.append(np.mean(train_ls))
        print("epoch:{} train_loss:{}".format(epoch, train_ls))
        with open(dir_name + "/log.txt", "a") as log:
            log.write("epoch:{} train_loss:{}\n".format(epoch, train_ls))
        if epoch % 10 == 0:
            model_name = dir_name + "/lstm_model_" + str(epoch) + ".net"
            with open(model_name, 'wb') as f:  # 训练完了保存模型
                torch.save(model.state_dict(), f)

    plt.plot(train_loss, label="train_loss")
    plt.title("loop vs epoch")
    plt.legend()
    plt.show()

    model_name = dir_name + "/lstm_model.net"

    with open(model_name, 'wb') as f:  # 训练完了保存模型
        torch.save(model.state_dict(), f)


def predict(model, char, top_k=1, hidden_size=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()  # 固定参数
    with torch.no_grad():
        char = np.array([char])  # 输入一个字符，预测下一个字是什么，先转成numpy
        char = char.reshape(-1, 1)  # 变成二维才符合编码规范
        char_encoding = model.onehot_encode(char).A  # 对char进行编码，取成numpy比较方便reshape
        char_encoding = char_encoding.reshape(1, 1, -1)  # char_encoding.shape为(1, 1, 43)变成三维才符合模型输入格式
        char_tensor = torch.tensor(char_encoding, dtype=torch.float32)  # 转成tensor
        char_tensor = char_tensor.to(device)
        out, hidden_size = model(char_tensor, hidden_size)  # 放入模型进行预测，out为结果
        probs = F.softmax(out, dim=1).squeeze()  # 计算预测值,即所有字符的概率
        probs, indices = probs.topk(top_k)
        indices = indices.cpu().numpy()
        probs = probs.cpu().numpy()
        char_index = np.random.choice(indices, p=probs / probs.sum())  # 随机选择一个字符索引作为预测值
        char = model.int2char[char_index]  # 通过索引找出预测字符

    return char, hidden_size


def sample(model, length, top_k=1, sentence="三"):
    hidden_size = None
    new_sentence = [char for char in sentence]
    for i in range(length):
        next_char, hidden_size = predict(model, new_sentence[-1], top_k=top_k, hidden_size=hidden_size)
        new_sentence.append(next_char)
    return "".join(new_sentence)


def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    if uchar in ('，', '。', '：', '？', '“', '”', '！', '；', '、', '《', '》', '——'):
        return True
    return False


def main():
    # 参数
    hidden_size = 512
    num_layers = 2
    batch_size = 128
    seq_len = 100
    epochs = 1000
    lr = 0.01

    dir_name = os.path.basename(__file__).split(".")[0]

    with open("data/三十三剑客图.txt", "r", encoding='gbk') as f:
        text = f.read()
        text = text.replace(
            '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        text = [word for word in text if is_uchar(word)]
        # text = [word for word in jieba.cut(text) if is_uchar(word)]
    vocab = np.array(sorted(set(text)))  # 建立字典

    trainset = np.array(list(text))

    model = lstm_model(vocab, hidden_size, num_layers)  # 模型实例化
    train(model, trainset, batch_size, seq_len, epochs, lr=lr)  # 训练模型
    model.load_state_dict(torch.load(dir_name + "/lstm_model_1300.net"))  # 调用保存的模型
    new_text = sample(model, 500, top_k=5,
                      sentence="唐朝开元年间")  # 预测模型，生成100个字符,预测时选择概率最大的前5个
    print(new_text)  # 输出预测文本


if __name__ == "__main__":
    main()
