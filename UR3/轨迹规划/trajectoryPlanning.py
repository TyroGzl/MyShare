"""
Time：2021/9/10 16:40
Version: V3.1.3
File: trajectoryPlanning.py
IDE:Jetbrains PyCharm
Description: 用于关节空间和笛卡尔空间轨迹规划的函数和实例
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline
from matplotlib import colors

'''
 *@brief:   关节空间规划函数
 *@date:    2021.9.10
 *@param:   wayPoint:途径的路径点，列表格式输入，waypoint要求至少有3个路径点，两个点没有规划的必要
 *@param：  durationTime：两个路径点间的运行时间，duration的长度要求比waypoint小1，需要是列表格式
 *@param:   acceleration：每个路径点抛物线过渡段期望的加速度，对正负没有要求，列表格式输入，有默认值50，不是必须参数
 *@param:   stopPoints：关节要停留的点，列表格式输入，不是必须参数
 *@param:   stopTime：停留点的停留时间，长度必须和stopPoints一致，列表格式输入，不是必须参数
 *@param:   mustPassPoint：机器人关节必须经过的路径点，列表格式，给点就行，不是必须参数
 *@returnParam: aOut：每个路径点过渡段加速度的大小
 *@returnParam: atOut：每个路径点加速度持续的时间
 *@returnParam: vOut：每段直线过渡段的速度大小
 *@returnParam: vtOut：每段直线过渡段直线运动的时间
'''


def tsPlanning(wayPoint, durationTime, **kwargs):  # 第一个参数如各路径点，第二个为各个区段的持续时间，kwargs里可以设置加速度的大小和需要精确经过的路径点
    #  先判断数据类型，确保数据类型不会出错
    if not isinstance(wayPoint, list):
        print("路径点请以集合的形式输入")
        return
    if not isinstance(durationTime, list):
        print("请以集合的形式输入各区段持续的时间")
        return
    if len(wayPoint) < 3:
        print("请至少输入3个路径点")
        return
    if len(wayPoint) - len(durationTime) != 1:
        print("路径点数和区段持续时间不匹配")
        return
    # 设置相关参数的默认值，然后检查相关参数是否被输入，若被输入则获取相应的值
    acceleration = 50  # 单位为°/s，默认值为50
    stopPoints = []
    stopTime = []
    mustPassPoint = []
    if 'acceleration' in kwargs:
        acceleration = kwargs['acceleration']
    while checkAcceleration(wayPoint, durationTime, acceleration):  # 确保该加速度可以求解出答案，不符合要求时按不长1迭代
        acceleration += 1
    if 'stopPoints' in kwargs:
        stopPoints = kwargs['stopPoints']
        if not 'stopTime' in kwargs:
            print("请输入驻留点停留的时间")
            return
        stopTime = kwargs['stopTime']
        if not isinstance(stopPoints, list):
            print("驻留点需要以列表的形式输入")
            return
        if not isinstance(stopTime, list):
            print('驻留时间需要以列表的形式输入')
            return
        j = 0
        for i in range(len(wayPoint)):
            if wayPoint[i + j] in stopPoints:
                wayPoint.insert(i + j, wayPoint[i + j])
                durationTime.insert(i + j, stopTime[stopPoints.index(wayPoint[i + j])])
                j += 1
    if 'mustPassPoint' in kwargs:
        mustPassPoint = kwargs['mustPassPoint']
    # 创建四个数组用来存储对应的结果
    velocity = []  # 对应书上的θjk.，直线运动速度
    transitionTime = []  # 对应书上的t,即过渡时间
    a = []  # 对应书上的θ..
    linearMotionTime = []  # 对应书上的tjk
    # 开始计算各个参数，先计算起始点
    a.append(np.sign(wayPoint[1] - wayPoint[0]) * abs(acceleration))
    transitionTime.append(durationTime[0] - np.sqrt(durationTime[0] ** 2 - (2 * (wayPoint[1] - wayPoint[0])) / a[0]))
    velocity.append((wayPoint[1] - wayPoint[0]) / (durationTime[0] - 0.5 * transitionTime[0]))
    for i in range(1, len(wayPoint) - 2):  # 直接先把所有的直线运动速度
        velocity.append((wayPoint[i + 1] - wayPoint[i]) / durationTime[i])
    for i in range(0, len(velocity) - 1):  # 计算相应的加速度,全部算完
        ac = np.sign(velocity[i + 1] - velocity[i]) * abs(acceleration)
        a.append(ac)
        if ac == 0:
            transitionTime.append(0)
        else:
            transitionTime.append((velocity[i + 1] - velocity[i]) / a[i + 1])
    if len(transitionTime) > 1:
        linearMotionTime.append(durationTime[0] - transitionTime[0] - 0.5 * transitionTime[1])
    # 再计算中间点
    for i in range(1, len(transitionTime) - 1):
        linearMotionTime.append(durationTime[i] - 0.5 * transitionTime[i] - 0.5 * transitionTime[i + 1])  # 计算线性运动时间
    # 计算终点
    lenWayPoint = len(wayPoint)
    # 最后这里要注意，倒数第二个和最后一个要联合起来求才行
    aFinal = np.sign(wayPoint[lenWayPoint - 2] - wayPoint[lenWayPoint - 1]) * acceleration
    transitionTimeFinal = durationTime[lenWayPoint - 2] - np.sqrt(
        durationTime[lenWayPoint - 2] ** 2 + 2 * (wayPoint[lenWayPoint - 1] - wayPoint[lenWayPoint - 2]) / aFinal)
    velocity.append((wayPoint[lenWayPoint - 1] - wayPoint[lenWayPoint - 2]) / (
            durationTime[lenWayPoint - 2] - 0.5 * transitionTimeFinal))
    a.append(np.sign(velocity[lenWayPoint - 2] - velocity[lenWayPoint - 3]) * abs(acceleration))
    transitionTime.append((velocity[lenWayPoint - 2] - velocity[lenWayPoint - 3]) / a[len(a) - 1])
    transitionTime.append(transitionTimeFinal)
    a.append(aFinal)
    linearMotionTime.append(
        durationTime[lenWayPoint - 3] - 0.5 * transitionTime[lenWayPoint - 2] - 0.5 * transitionTime[lenWayPoint - 3])
    linearMotionTime.append(
        durationTime[lenWayPoint - 2] - transitionTime[lenWayPoint - 1] - 0.5 * transitionTime[lenWayPoint - 2])
    aOut = []
    atOut = []
    vOut = []
    vtOut = []
    print("各过渡段的加速度为：")
    for i in range(len(a)):
        aOut.append(a[i])
        print(a[i], end=' ')
    print("\n各过渡段时间为：")
    for i in range(len(transitionTime)):
        print(transitionTime[i], end=' ')
        atOut.append(transitionTime[i])
    print("\n各直线运动速度为：")
    for i in range(len(velocity)):
        print(velocity[i], end=' ')
        vOut.append(velocity[i])
    print("\n各区段直线运动的时间为：")
    for i in range(len(linearMotionTime)):
        print(linearMotionTime[i], end=' ')
        vtOut.append(linearMotionTime[i])
    return aOut, atOut, vOut, vtOut


'''
 *@brief:   检查加速度是否符合要求的函数
 *@date:    2021.9.10
 *@param:   wayPoints1:途径的路径点，列表格式输入，wayPoints1要求至少有3个路径点，两个点没有规划的必要
 *@param：  durationTime1：两个路径点间的运行时间，duration的长度要求比waypoint小1，需要是列表格式
 *@param:   acceleration：每个路径点抛物线过渡段期望的加速度，对正负没有要求，列表格式输入，有默认值50，不是必须参数
 *@returnParam: 1表示符合要求，0为不符合要求 
'''


def checkAcceleration(wayPoints1, durationTime1, acceleration1):  # 检查加速度是否符合要求
    flag = 0
    for i in range(len(durationTime1)):
        if acceleration1 >= 4 * (wayPoints1[i + 1] - wayPoints1[i]) / durationTime1[i] ** 2:
            continue
        else:
            flag = 1
            break
    if flag == 1:
        return 1
    else:
        return 0


'''
 *@brief:   显示关节空间规划后的点的函数
 *@date:    2021.9.11
 *@param:   accelerationf：每个路径点抛物线过渡段的加速度
 *@param:   accelerationTimef：每个抛物线过渡段加速度持续的时间
 *@param:   velocityf：每个直线段路径的运动速度
 *@param:   velocicyTimef：每个直线段运动持续的时间
 *@param:   stepf：时间步长，越小插值点越多
 *@param:   wayPointf：指定的路径点，主要是用来获得起始点的
 *@param:   durationTimef：给定的每个路径段的经历时间
'''


def showData(accelerationf, accelerationTimef, velocityf, velocityTimef, stepf, wayPointf, durationTimef):
    t = 0
    totalTime = 0
    currentState = 0  # 0表示在加速阶段
    x = [0]
    y = [wayPointf[0]]
    xita = wayPointf[0]
    aIndex = 0
    vIndex = 0
    a0 = 0
    a1 = 0
    a2 = 0
    stageTime = accelerationTimef[0]
    lastStageTime = 0
    print("\n\n接下来输出多项式的系数")
    for i in durationTimef:
        totalTime += i
    while t < totalTime:
        if currentState == 0:  # 连接各段函数表达式
            if aIndex != 0:  # 中间点需要补偿高度
                a2 = accelerationf[aIndex] / 2
                if aIndex == len(accelerationf) - 1:  # 终点右端速度为0
                    a1 = -2 * a2 * stageTime
                else:
                    a1 = velocityf[vIndex] - 2 * a2 * stageTime
                a0 = xita - a1 * t - a2 * t ** 2
                print('过渡段的系数为x=a0+a1t+a2t^2： 区间为', end=' ')
                print(lastStageTime, stageTime)
                print(a0, a1, a2)
            else:
                print('起始点表达式系数为x=at^2： 区间为', end=' ')
                print(lastStageTime, stageTime)
                print(0.5 * accelerationf[aIndex])
        else:
            print('直线段系数为x=a0+a1t： 区间为', end=' ')
            print(lastStageTime, stageTime)
            print(xita - velocityf[vIndex - 1] * lastStageTime, velocityf[vIndex - 1])
        while t < stageTime:
            t += stepf
            x.append(t)
            if currentState == 0:
                if aIndex == 0:
                    y.append(xita + 0.5 * accelerationf[aIndex] * t ** 2)
                else:
                    y.append(a0 + a1 * t + a2 * t ** 2)
            else:
                y.append(xita + velocityf[vIndex - 1] * (t - lastStageTime))
        xita = y[len(y) - 1]
        if t > totalTime:
            break
        if currentState == 0:
            lastStageTime = stageTime
            stageTime += velocityTimef[vIndex]
            vIndex += 1
            currentState = 1
        else:
            aIndex += 1
            lastStageTime = stageTime
            t = stageTime
            stageTime += accelerationTimef[aIndex]
            currentState = 0
    plt.figure(figsize=[4, 3])
    plt.subplots_adjust(right=0.95, top=0.9, left=0.18, bottom=0.15, wspace=0.3, hspace=0.3)

    plt.plot(x, y, label='关节角度曲线')
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.title('关节角随时间的变化曲线', fontsize=14)
    plt.xlabel('时间/s', fontsize=14)
    plt.ylabel('关节角/°', fontsize=14)
    plt.legend()
    plt.show()


'''
 *@brief:   笛卡尔空间直线规划函数
 *@date:    2021.9.12
 *@param:   p1f：起始点坐标，元组形式输入
 *@param:   p2f：终止点坐标，元组形式输入
 *@param:   dotNum：插值点数，以int形式输入
 *@returnParam:   pointf：规划后的点坐标list
'''


def linearPlanning(p1f, p2f, dotNum):
    if not (isinstance(p1f, tuple) and isinstance(p2f, tuple)):
        print('两个路径点需要以元组的形式输入')
        return
    if len(p1f) != 3 or len(p2f) != 3:
        print('p1,p2中存在某点元组长度不为3，请检查')
        return
    deltax = (p2f[0] - p1f[0]) / dotNum
    deltay = (p2f[1] - p1f[1]) / dotNum
    deltaz = (p2f[2] - p1f[2]) / dotNum
    pointf = []  # 存储路径点
    currentIndex = 0
    while currentIndex <= dotNum:
        x = p1f[0] + deltax * currentIndex
        y = p1f[1] + deltay * currentIndex
        z = p1f[2] + deltaz * currentIndex
        pointf.append((x, y, z))
        currentIndex += 1
    return pointf


'''
 *@brief:   根据空间三点求圆心半径的函数
 *@date:    2021.9.11
 *@param:   p1f：空间中三个点中的一个，元组形式输入
 *@param:   p2f：空间中三个点中的一个，元组形式输入
 *@param:   p3f：空间中三个点中的一个，元组形式输入
 *@returnParam: centerf：圆心坐标，元组形式输出
 *@returnParam: rad：空间圆半径
'''


def getCenterRad(p1f, p2f, p3f):  # 给定空间中三个点，计算圆心半径的函数
    # 先检查格式
    if not (isinstance(p1f, tuple) and isinstance(p2f, tuple) and isinstance(p3f, tuple)):
        print('三个路径点需要以元组的形式输入')
        return
    if len(p1f) != 3 or len(p2f) != 3 or len(p3f) != 3:
        print('p1,p2,p3中存在某点元组长度不为3，请检查')
        return
    # 计算p1到p2和p1到p3的单位向量
    p1v = np.array(p1f)
    p2v = np.array(p2f)
    p3v = np.array(p3f)
    v1f = p2v - p1v
    v2f = p3v - p1v
    if np.linalg.norm(v1f) == 0 or np.linalg.norm(v2f) == 0:
        print("输入点不能相同")
        return
    v1nf = v1f / np.linalg.norm(v1f)
    v2nf = v2f / np.linalg.norm(v2f)
    # 检查三点是否共线
    nvf = np.cross(v1nf, v2nf)
    if np.all(nvf == 0):  # 说明三点共线，不符合要求
        print('请确保三点不共线')
    '''
    if find(sum(abs(nv),2)<1e-5)
    fprintf('三点过于趋近直线\n');rad = -1;return;
    '''
    # 计算心坐标系的UVW轴
    uf = v1nf
    wf = np.cross(v2f, v1f) / np.linalg.norm(np.cross(v2f, v1f))
    vf = np.cross(wf, uf)
    # 计算投影
    bxf = np.dot(v1f, uf)
    cxf = np.dot(v2f, uf)
    cyf = np.dot(v2f, vf)
    # 计算圆心
    hf = ((cxf - bxf / 2) ** 2 + cyf ** 2 - (bxf / 2) ** 2) / (2 * cyf)
    c2 = np.dot(bxf / 2, uf)
    c3 = np.dot(hf, vf)
    centerfx = p1f[0] + c2[0] + c3[0]
    centerfy = p1f[1] + c2[1] + c3[1]
    centerfz = p1f[2] + c2[2] + c3[2]
    centerf = (centerfx, centerfy, centerfz)
    rad = np.sqrt((centerf[0] - p1f[0]) ** 2 + (centerf[1] - p1f[1]) ** 2 + (centerf[2] - p1f[2]) ** 2)
    return centerf, rad  # 返回圆心和半径


'''
 *@brief:   笛卡尔空间圆轨迹规划的函数
 *@date:    2021.9.11
 *@param:   p1f：空间中三个点中的一个，元组形式输入
 *@param:   p2f：空间中三个点中的一个，元组形式输入
 *@param:   p3f：空间中三个点中的一个，元组形式输入
 *@param:   centerf：空间圆的圆心，元组形式输入
 *@param:   rf：空间圆的半径
 *@param:   interpolationNum：插值点数，int类型
 *@returnParam: pointf：规划出来的轨迹的点集，列表格式，点以元组形式存储
'''


def circularPlanning(p1f, p2f, p3f, centerf, rf, interpolationNum):  # 点需要以元组的形式输入
    # 先检查格式
    if not (isinstance(p1f, tuple) and isinstance(p2f, tuple) and isinstance(p3f, tuple)):
        print('三个路径点需要以元组的形式输入')
        return
    if len(p1f) != 3 or len(p2f) != 3 or len(p3f) != 3:
        print('p1,p2,p3中存在某点元组长度不为3，请检查')
        return
    # 计算p1到p2和p1到p3的单位向量
    p1v = np.array(p1f)
    p2v = np.array(p2f)
    p3v = np.array(p3f)
    pcv = np.array(centerf)
    # 计算A,B,C
    A = (p2f[1] - p1f[1]) * (p3f[2] - p2f[2]) - (p2f[2] - p1f[2]) * (p3f[1] - p2f[1])
    B = (p2f[2] - p1f[2]) * (p3f[0] - p2f[0]) - (p2f[0] - p1f[0]) * (p3f[2] - p2f[2])
    C = (p2f[0] - p1f[0]) * (p3f[1] - p2f[1]) - (p2f[1] - p1f[1]) * (p3f[0] - p2f[0])
    K = np.sqrt(A ** 2 + B ** 2 + C ** 2)
    # 计算a向量
    a = np.array([A / K, B / K, C / K])
    n = (p1v - pcv) / rf
    o = np.cross(a, n)
    T = np.row_stack(
        (np.column_stack([n.transpose(), o.transpose(), a.transpose(), pcv.transpose()]), np.array([0, 0, 0, 1])))
    # 求转换后的点
    q1f = np.dot(np.linalg.inv(T), np.concatenate((p1v.transpose(), np.array([1])), axis=0).transpose())
    q2f = np.dot(np.linalg.inv(T), np.concatenate((p2v.transpose(), np.array([1])), axis=0).transpose())
    q3f = np.dot(np.linalg.inv(T), np.concatenate((p3v.transpose(), np.array([1])), axis=0).transpose())
    # 计算角度
    if q3f[1] < 0:
        theta13 = math.atan2(q3f[1], q3f[0]) + 2 * math.pi
    else:
        theta13 = math.atan2(q3f[1], q3f[0])

    if q2f[1] < 0:
        theta12 = math.atan2(q2f[1], q2f[0]) + 2 * math.pi
    else:
        theta12 = math.atan2(q2f[1], q2f[0])

    # 轨迹插补
    pointf = []  # 存储插补点
    currentTheta = 0
    thetaStep = theta13 / interpolationNum
    while currentTheta <= theta13:
        pointf.append(
            np.delete(np.dot(T, np.hstack([rf * np.cos(currentTheta), rf * np.sin(currentTheta), 0, 1]).transpose()), 3,
                      axis=0).tolist())
        currentTheta += thetaStep
    return pointf


'''
 *@brief:   显示空间轨迹的函数
 *@date:    2021.9.12
 *@param:   pointf：规划出来的空间点
 *@param:   waypointf：给定的路径点
 *@returnParam: 无，最后会显示图像
'''


def showSpacePath(pointf, waypointf):
    xf = []
    yf = []
    zf = []
    xof = []
    yof = []
    zof = []
    color = []
    cMax = 1
    cMin = 0.9
    j = 0
    for i in pointf:
        xf.append(i[0])
        yf.append(i[1])
        zf.append(i[2])
        color.append(cMin + (cMax - cMin) * j / len(pointf))
        j += 1
    for i in waypointf:
        xof.append(i[0])
        yof.append(i[1])
        zof.append(i[2])
    fig = plt.figure()
    # plt.scatter(x, y)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(xf, yf, zf, c=color, alpha=0.3, cmap='summer', label='planned points')
    ax.scatter(xof, yof, zof, c='r', label='original points', s=70)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'black'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'black'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'black'})
    ax.legend()
    plt.show()


'''
 *@brief:   笛卡尔空间一般曲线轨迹规划的函数
 *@date:    2021.9.13
 *@param:   wayPointf：给定的路径点，列表类型，点以元组存在列表中
 *@param:   timef：到达每个路径点时的时间，以起始点为时间零点，列表类型
 *@param:   accurate：准确度参数，非必须，默认准确经过每个路径点，accurate越大越不准确，但曲线会越光滑，int型
 *@param:   loop：循环参数，非必须，1表示该路径要循环执行，其他值表示不需要循环执行，int型
 *@param:   dotNum：插值点数量，非必须，默认50，int类型
 *@param:   interpolationNum：插值点数，int类型
 *@returnParam: pointf：规划出来的轨迹的点集，列表格式，点以元组形式存储
'''


def generalCurvePlanning(wayPointf, timef, **kwargs):  # 默认平滑度为1
    smoothf = 0
    loop = 0
    dotNumf = 50
    # 先检查数据格式和获取输入的参数
    if not isinstance(wayPointf, list) or len(wayPointf) == 0:
        print('路径点数据需要以列表的形式输入，且列表非空')
        return
    if 'accurate' in kwargs:
        smoothf = kwargs['accurate']
    if 'loop' in kwargs:
        loop = kwargs['loop']
    if 'dotNum' in kwargs:
        dotNumf = kwargs['dotNum']
    if not isinstance(timef, list):
        print('输入的时间数据必须是列表格式')
        return
    if loop == 1:
        if len(timef) != len(wayPointf) + 1:
            print('当循环执行路径时，要求时间列表的长度比路径点列表长度多1')
            return
    else:
        if len(timef) != len(wayPointf):
            print('当不循环执行路径时，要求时间列表的长度等于路径点列表长度')
            return
    # 分别获取x，y，z和验证输入的数据的格式
    xf = []
    yf = []
    zf = []
    tf = timef
    for instancef in wayPointf:
        if not isinstance(instancef, tuple) or len(instancef) != 3:
            print('列表里的数据需要以三维元组Tuple的形式存在')
            return
        xf.append(instancef[0])
        yf.append(instancef[1])
        zf.append(instancef[2])
    # 检查是否需要循环执行
    if loop == 1:
        xf.append(xf[0])
        yf.append(yf[0])
        zf.append(zf[0])
        xf.append(xf[1])
        yf.append(yf[1])
        zf.append(zf[1])
        tf.append(timef[len(timef) - 1] + (timef[1] - timef[0]))
    # 开始进行平滑拟合
    if loop == 1:
        tsf = np.linspace(tf[0], timef[len(timef) - 2], dotNumf)
    else:
        tsf = np.linspace(tf[0], timef[len(timef) - 1], dotNumf)
    s1 = UnivariateSpline(tf, xf, s=smoothf)
    xsf = s1(tsf)

    s2 = UnivariateSpline(tf, yf, s=smoothf)
    ysf = s2(tsf)

    s3 = UnivariateSpline(tf, zf, s=smoothf)
    zsf = s3(tsf)
    pointf = []
    for i in range(len(xsf)):
        pointf.append((xsf[i], ysf[i], zsf[i]))
    return pointf


if __name__ == '__main__':
    # 关节空间规划举例
    # wayPoints = [0, -35, -20, -50, -50, 0]  # 这里添加路径点
    # durationTimes = [2, 1, 2, 1, 3]  # 这里添加路径段持续的时间
    # (ap, atp, vp, vtp) = tsPlanning(wayPoints, durationTimes, acceleration=20)
    # # 接下来开始计算各个路径点并画图
    # steps = 0.0001
    # showData(ap, atp, vp, vtp, steps, wayPoints, durationTimes)

    '''
    # 笛卡尔空间直线规划举例
    p1 = (0, 0, 0)
    p2 = (2, 4, 5)
    dotNum = 50
    # 以下为程序运行段，不用改动，只需要改动上面的两个点坐标和插入点数即可
    pointss = linearPlanning(p1, p2, dotNum)
    showSpacePath(pointss, [p1, p2)
    '''

    # 笛卡尔空间圆轨迹规划举例
    p1 = (-118.43 / 1000, -268.05 / 1000, 157.28 / 1000)
    p2 = (-209.5 / 1000, -209.67 / 1000, 291.44 / 1000)
    p3 = (-157.76 / 1000, -278.26 / 1000, 143.82 / 1000)
    dotSum = 500
    # 以下为程序执行段，不需要改动，只需要改动上面的三个点和插值点
    (center, rad) = getCenterRad(p1, p2, p3)
    print(center, rad)
    pointss = circularPlanning(p1, p2, p3, center, rad, dotSum)
    showSpacePath(pointss, [p1, p2, p3])
    print(pointss)

    '''
    # 笛卡尔空间一般曲线规划举例
    # 当不要求循环执行时
    # wayPoints=[(0, 0, 0), (2, 1, 2), (2, 4, 5), (4, 5, 6), (2, 6, 2)]
    # times = [0, 1, 4, 6, 8]  # 此时时间长度应和路径点相同
    # pointss = generalCurvePlanning(wayPoints, times, accurate=0, loop=0)  # 默认不循环执行,默认经过所有路径点，默认插入点数为50
    # showSpacePath(pointss, wayPoints)
    '''
    # 笛卡尔空间一般曲线规划举例
    # 当要求循环执行时
    '''
    wayPoints = [(0, 0, 0), (2, 1, 2), (2, 4, 5), (4, 5, 6), (2, 6, 2)]
    times = [0, 1, 4, 6, 8, 12]  # 此时时间长度应比路径点多1
    dotNum = 100
    pointss = generalCurvePlanning(wayPoints, times, accurate=0, loop=1, dotNum=dotNum)  # 默认不循环执行,默认经过所有路径点，默认插入点数为50
    showSpacePath(pointss, wayPoints)
    '''
