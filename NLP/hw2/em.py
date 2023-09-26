import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math


class GMM:
    def __init__(self, data, n, theta=None, miu=None, sigma=None):
        self.data = data
        self.data_len = len(data)
        self.n = n
        if theta is not None:
            self.theta = theta
        else:
            self.theta = [1 / self.n for i in range(self.n)]

        if miu is not None:
            self.miu = miu
        else:
            self.miu = []
            for i in range(self.n):
                sample_count = int(len(self.data) * self.theta[i])
                sample = random.sample(self.data, sample_count)
                self.miu.append(sum(sample) / sample_count)

        if sigma is not None:
            self.sigma = sigma
        else:
            self.sigma = [1 for i in range(self.n)]

    def gaussian(self, x, miu, sigma):
        result = (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-0.5 * ((x - miu) / sigma) ** 2)
        return result

    def gmm_em(self, iter_max):
        iter = 0
        eps = 1e-10
        while True:
            iter += 1
            # e步
            gamma = np.zeros((self.data_len, self.n))
            for i in range(self.data_len):
                for n in range(self.n):
                    gamma[i][n] = self.theta[n] * self.gaussian(self.data[i], self.miu[n], self.sigma[n])
                gamma[i] = gamma[i] / sum(gamma[i])
            # m步
            miu_new = [0, 0]
            sigma_new = [0, 0]
            theta_new = [0, 0]
            for n in range(self.n):
                miu_new[n] = np.dot(self.data, gamma[:, n]) / np.sum(gamma[:, n])
                sigma_new[n] = math.sqrt(np.dot((np.array(self.data) - miu_new[n]) ** 2, gamma[:, n]) / np.sum(
                    gamma[:, n]))
                theta_new[n] = np.mean(gamma[:, n])
            if np.max(np.array(miu_new) - np.array(self.miu)) < eps and np.max(
                    np.array(sigma_new) - np.array(self.sigma)) < eps and np.max(
                np.array(theta_new) - np.array(self.theta)) < eps:
                self.miu = miu_new
                self.sigma = sigma_new
                self.theta = theta_new
                break
            else:
                self.miu = miu_new
                self.sigma = sigma_new
                self.theta = theta_new


if __name__ == '__main__':
    data_read = pd.read_csv('height_data.csv')
    data = []
    for i in range(data_read.size):
        data.append(data_read.values[i].tolist()[0])
    g = GMM(data, 2, theta=[0.5, 0.5], miu=[170, 170], sigma=[1, 1])
    g.gmm_em(1000)
    print(
        "男生比例为{:.4f}，相对偏差为{:.2%}，男生身高均值为{:.4f}，相对偏差为{:.2%}，男生身高方差为{:.4f}，相对偏差为{:.2%}".format(
            g.theta[0], (g.theta[0] - 0.75) / 0.75, g.miu[0], (g.miu[0] - 176) / 176, g.sigma[0], (g.sigma[0] - 5) / 5))
    print(
        "女生比例为{:.4f}，相对偏差为{:.2%}，女生身高均值为{:.4f}，相对偏差为{:.2%}，女生身高方差为{:.4f}，相对偏差为{:.2%}".format(
            g.theta[1], (g.theta[1] - 0.25) / 0.25, g.miu[1], (g.miu[1] - 164) / 164, g.sigma[1], (g.sigma[1] - 3) / 3))

    g = GMM(data, 2, theta=[0.5, 0.5], miu=[170, 160], sigma=[1, 1])
    g.gmm_em(1000)
    print(
        "男生比例为{:.4f}，相对偏差为{:.2%}，男生身高均值为{:.4f}，相对偏差为{:.2%}，男生身高方差为{:.4f}，相对偏差为{:.2%}".format(
            g.theta[0], (g.theta[0] - 0.75) / 0.75, g.miu[0], (g.miu[0] - 176) / 176, g.sigma[0], (g.sigma[0] - 5) / 5))
    print(
        "女生比例为{:.4f}，相对偏差为{:.2%}，女生身高均值为{:.4f}，相对偏差为{:.2%}，女生身高方差为{:.4f}，相对偏差为{:.2%}".format(
            g.theta[1], (g.theta[1] - 0.25) / 0.25, g.miu[1], (g.miu[1] - 164) / 164, g.sigma[1], (g.sigma[1] - 3) / 3))
