# -*- coding:utf8 -*-
import math
import matplotlib.pyplot as plt
import numpy
def feature_whole(x1, x2, e):
    #d = [1,x1,x2]
    d = [1]
    for n in range(1,e+1):
        for i in range(n+1):
            d.append(pow(x1, n-i) * pow(x2, i))
    return d
def h(w, d):
    n = len(w)
    x = 0
    for i in range(n):
        x = x + w[i]*d[i]
    if(x > 500):
        return 1
    if(x < -500):
        return 0
    return 1/(1+math.exp(-x))
def logistic_regression(data, alpha, lamda):
    n = len(data[0]) - 1
    w = [0 for x in range(n)]
    w2 = [0 for x in range(n)]
    for times in range(10000):
        for d in data:
            for i in range(n):
                w2[i] = w[i] + alpha * (d[n] - h(w, d))*d[i] + lamda * w[i]
            for i in range(n):
                w[i] = w2[i]
        print times, w
    return w
def logistic_regression2(data, alpha, lamda):
    n = len(data[0]) - 1
    w = [0 for x in range(n)]
    for times in range(1000):
        for d in data:
            for i in range(n):
                w[i] = w[i] + alpha * (d[n] - h(w, d))*d[i] + lamda * w[i]
        print w
    return w
def Min2(a, b):
    m = min(a)
    m2 = min(b)
    if(m < m2):
        return m
    return m2
def Max2(a, b):
    m = max(a)
    m2 = max(b)
    if(m > m2):
        return m
    return m2
def show_data(data, w, e):
    m = len(data)
    n = len(data[0])
    nx1 = []
    nx2 = []
    px1 = []
    px2 = []
    # 样本点
    for d in data:
        if d[n-1] == 0:
            nx1.append(d[1])
            nx2.append(d[2])
        else:
            px1.append(d[1])
            px2.append(d[2])
    #划分区域
    min1 = Min2(nx1, px1)
    min2 = Min2(nx2, px2)
    max1 = Max2(nx1, px1)
    max2 = Max2(nx2, px2)
    r_nx1 = []
    r_nx2 = []
    r_px1 = []
    r_px2 = []
    step1 = (max1 - min1) / 300
    step2 = (max2 - min2) / 300
    for x1 in numpy.arange(min1, max1, step1):
        for x2 in numpy.arange(min2, max2, step2):
            d = feature_whole(x1, x2, e)
            if h(w, d) > 0.5:
                r_px1.append(x1)
                r_px2.append(x2)
            else:
                r_nx1.append(x1)
                r_nx2.append(x2)
    plt.plot(r_px1, r_px2, 'c.')
    plt.plot(r_nx1, r_nx2, 'y.')
    plt.plot(px1, px2, 'ro', markersize=7)
    plt.plot(nx1, nx2, 'bo', markersize=7)
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    e = 2
    fileData = open("data.txt")
    data = []
    for line in fileData:
        d = map(float, line.split(','))
        d2 = feature_whole(d[1], d[2], e)
        d2.append(d[3])
        print d2
        data.append(d2)
    fileData.close()
    alpha = 0.001
    lamda = 0.00001
    w = logistic_regression(data, alpha, lamda)
    print w
    show_data(data, w, e)
