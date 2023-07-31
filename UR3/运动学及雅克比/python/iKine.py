import math


def cos(q):
    return math.cos(q)


def sin(q):
    return math.sin(q)


def sqrt(q):
    return math.sqrt(q)


def atan2(x, y):
    return math.atan2(x, y)


def acos(x):
    return math.acos(x)


def ikine(pose, last_q):
    """
    使用旋转向量求解逆运动学

    :param pose: 位姿，输入 1 x 6 列表
    :param last_q: 上一时刻关节角，输入 1 x 6 列表
    :return: 关节角度
    """
    eps = 1e-5
    # UR3机器人参数
    a2 = -0.24365
    a3 = -0.213
    d1 = 0.1519
    d2 = 0.11985
    d4 = -0.00945
    d5 = 0.0834
    d6 = 0.0824
    # 旋转向量转为旋转矩阵
    px = pose[0]
    py = pose[1]
    pz = pose[2]
    r1 = pose[3]
    r2 = pose[4]
    r3 = pose[5]
    theta = sqrt(r1 ** 2 + r2 ** 2 + r3 ** 2)
    kx = r1 / theta
    ky = r2 / theta
    kz = r3 / theta
    nx = kx * kx * (1 - cos(theta)) + cos(theta)
    ox = kx * ky * (1 - cos(theta)) - kz * sin(theta)
    ax = kx * kz * (1 - cos(theta)) + ky * sin(theta)
    ny = kx * ky * (1 - cos(theta)) + kz * sin(theta)
    oy = ky * ky * (1 - cos(theta)) + cos(theta)
    ay = ky * kz * (1 - cos(theta)) - kx * sin(theta)
    nz = kx * kz * (1 - cos(theta)) - ky * sin(theta)
    oz = ky * kz * (1 - cos(theta)) + kx * sin(theta)
    az = kz * kz * (1 - cos(theta)) + cos(theta)
    # 开始求解
    m = py - ay * d6
    n = px - ax * d6

    count = 1  # 解的序号
    bios = 0
    minBios = 1e8
    q = []
    output = []
    weights = [0.5, 0.01, 0.5, 0.01, 0.5, 0.01]

    for index_i in [1, 2]:
        if (m ** 2 + n ** 2 - (d2 + d4) ** 2) < 0:
            count = count + 1
            continue

        theta_1 = atan2(m, n) - atan2(-d2 - d4, (-1) **
                                      index_i * sqrt(m ** 2 + n ** 2 - (d2 + d4) ** 2))

        if abs(theta_1) < eps:
            theta_1 = 0

        for index_j in [1, 2]:
            c5 = ax * sin(theta_1) - ay * cos(theta_1)

            if abs(c5) > 1:
                # print('第{0}组解为：\n'.format(count))
                # print('无解\n')
                count = count + 1
                continue

            theta_5 = (-1) ** index_j * acos(c5)

            if abs(theta_5) < eps:
                theta_5 = 0

            s = cos(theta_1) * ny - sin(theta_1) * nx
            t = cos(theta_1) * oy - sin(theta_1) * ox

            if theta_5 == 0:
                theta_6 = last_q[5]
            else:
                theta_6 = atan2(s, t) - atan2(-sin(theta_5), 0)

            if abs(theta_6) < eps:
                theta_6 = 0

            r14 = px * cos(theta_1) + sin(theta_1) * py - d6 * (sin(theta_1) * ay + cos(theta_1) * ax) \
                  + d5 * (cos(theta_1) * cos(theta_6) * ox + cos(theta_1) * sin(theta_6) * nx + cos(theta_6)
                          * sin(theta_1) * oy + sin(theta_1) * sin(theta_6) * ny)
            r34 = pz - d1 - az * d6 + d5 * (sin(theta_6) * nz + cos(theta_6) * oz)

            for index_k in [1, 2]:
                c3 = (r14 ** 2 + r34 ** 2 - a3 **
                      2 - a2 ** 2) / (2 * a2 * a3)

                if abs(c3) > 1:
                    # print('第{0}组解为：\n'.format(count))
                    # print('无解\n')
                    count = count + 1
                    continue

                theta_3 = (-1) ** index_k * acos(c3)
                if abs(theta_3) < eps:
                    theta_3 = 0

                s2 = (r34 * (a3 * cos(theta_3) + a2) - a3 * sin(theta_3)
                      * r14) / (a3 ** 2 + a2 ** 2 + 2 * a2 * a3 * cos(theta_3))
                c2 = (r34 * a3 * sin(theta_3) + (a3 * cos(theta_3) + a2)
                      * r14) / (a3 ** 2 + a2 ** 2 + 2 * a2 * a3 * cos(theta_3))
                theta_2 = atan2(s2, c2)

                if abs(theta_2) < eps:
                    theta_2 = 0

                s234 = -sin(theta_6) * (cos(theta_1) * nx + sin(theta_1) * ny) \
                       - cos(theta_6) * (cos(theta_1) * ox + sin(theta_1) * oy)
                c234 = sin(theta_6) * nz + cos(theta_6) * oz
                theta_4 = atan2(s234, c234) - theta_2 - theta_3

                if abs(theta_4) < eps:
                    theta_4 = 0

                q = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]
                count = count + 1
                for index in range(6):
                    bios += (q[index] - last_q[index]) ** 2 * weights[index]
                if bios < minBios:
                    minBios = bios
                    output = q
                    bios = 0

                # print('第{0}组解为：\n'.format(count))
                # print(theta)

    return output
