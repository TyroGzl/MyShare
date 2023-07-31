import math
# from pprint import pprint

import numpy as np
# from sympy import pprint, Symbol, cos, sin

# UR3机器人参数
a2 = -243.65
a3 = -213
d1 = 151.9
d2 = 119.85
d4 = -9.45
d5 = 83.4
d6 = 82.4


def cos(q):
    return math.cos(q)


def sin(q):
    return math.sin(q)


def myJacob(q):
    """
    计算{0}系下的雅克比矩阵

    :param q: 关节角（弧度制）
    :return: 雅可比矩阵
    """
    J11 = d6 * (cos(q[0]) * cos(q[4]) + cos(q[1] + q[2] + q[3]) * sin(q[0]) * sin(q[4])) + d2 * cos(q[0]) + d4 * cos(
        q[0]) - a3 * cos(
        q[1] + q[2]) * sin(q[0]) - a2 * cos(q[1]) * sin(q[0]) - d5 * sin(q[1] + q[2] + q[3]) * sin(q[0])
    J21 = d6 * (cos(q[4]) * sin(q[0]) - cos(q[1] + q[2] + q[3]) * cos(q[0]) * sin(q[4])) + d2 * sin(q[0]) + d4 * sin(
        q[0]) + a3 * cos(
        q[1] + q[2]) * cos(q[0]) + a2 * cos(q[0]) * cos(q[1]) + d5 * sin(q[1] + q[2] + q[3]) * cos(q[0])
    J31 = 0
    J41 = 0
    J51 = 0
    J61 = 1

    J12 = -cos(q[0]) * (a3 * sin(q[1] + q[2]) + a2 * sin(q[1]) - d5 *
                        cos(q[1] + q[2] + q[3]) - d6 * sin(q[1] + q[2] + q[3]) * sin(q[4]))
    J22 = -sin(q[0]) * (a3 * sin(q[1] + q[2]) + a2 * sin(q[1]) - d5 *
                        cos(q[1] + q[2] + q[3]) - d6 * sin(q[1] + q[2] + q[3]) * sin(q[4]))
    J32 = a3 * cos(q[1] + q[2]) + a2 * cos(q[1]) + d5 * (
                cos(q[1] + q[2]) * sin(q[3]) + sin(q[1] + q[2]) * cos(q[3])) - d6 * sin(q[4]) * (
                  cos(q[1] + q[2]) * cos(q[3]) - sin(q[1] + q[2]) * sin(q[3]))
    J42 = sin(q[0])
    J52 = -cos(q[0])
    J62 = 0

    J13 = cos(q[0]) * (d5 * cos(q[1] + q[2] + q[3]) - a3 *
                       sin(q[1] + q[2]) + d6 * sin(q[1] + q[2] + q[3]) * sin(q[4]))
    J23 = sin(q[0]) * (d5 * cos(q[1] + q[2] + q[3]) - a3 *
                       sin(q[1] + q[2]) + d6 * sin(q[1] + q[2] + q[3]) * sin(q[4]))
    J33 = a3 * cos(q[1] + q[2]) + d5 * sin(q[1] + q[2] + q[3]) - \
          d6 * cos(q[1] + q[2] + q[3]) * sin(q[4])
    J43 = sin(q[0])
    J53 = -cos(q[0])
    J63 = 0

    J14 = cos(q[0]) * (d5 * cos(q[1] + q[2] + q[3]) +
                       d6 * sin(q[1] + q[2] + q[3]) * sin(q[4]))
    J24 = sin(q[0]) * (d5 * cos(q[1] + q[2] + q[3]) +
                       d6 * sin(q[1] + q[2] + q[3]) * sin(q[4]))
    J34 = d5 * sin(q[1] + q[2] + q[3]) - d6 * \
          cos(q[1] + q[2] + q[3]) * sin(q[4])
    J44 = sin(q[0])
    J54 = -cos(q[0])
    J64 = 0

    J15 = -d6 * (sin(q[0]) * sin(q[4]) + cos(q[1] +
                                             q[2] + q[3]) * cos(q[0]) * cos(q[4]))
    J25 = d6 * (cos(q[0]) * sin(q[4]) - cos(q[1] +
                                            q[2] + q[3]) * cos(q[4]) * sin(q[0]))
    J35 = -d6 * sin(q[1] + q[2] + q[3]) * cos(q[4])
    J45 = sin(q[1] + q[2] + q[3]) * cos(q[0])
    J55 = sin(q[1] + q[2] + q[3]) * sin(q[0])
    J65 = -cos(q[1] + q[2] + q[3])

    J16 = 0
    J26 = 0
    J36 = 0
    J46 = cos(q[4]) * sin(q[0]) - cos(q[1] +
                                      q[2] + q[3]) * cos(q[0]) * sin(q[4])
    J56 = - cos(q[0]) * cos(q[4]) - cos(q[1] +
                                        q[2] + q[3]) * sin(q[0]) * sin(q[4])
    J66 = -sin(q[1] + q[2] + q[3]) * sin(q[4])

    J = np.mat([[J11, J12, J13, J14, J15, J16], [J21, J22, J23, J24, J25, J26], [J31, J32, J33, J34, J35, J36],
                [J41, J42, J43, J44, J45, J46], [J51, J52, J53, J54, J55, J56], [J61, J62, J63, J64, J65, J66]])
    eps = 1e-5
    for i in range(6):
        for j in range(6):
            if abs(J[i, j]) < eps:
                J[i, j] = 0
    return J


if __name__ == '__main__':
    P = 180 / math.pi
    J = myJacob([10 / P, 20 / P, 30 / P, 40 / P, 50 / P, 60 / P])
    print(J)
