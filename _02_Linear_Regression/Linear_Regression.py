# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    x, y = read_data()
    w = np.dot(np.linalg.inv(np.dot(x.T, x) + 0.5 * np.eye(6)), np.dot(x.T, y))
    return w @ data




def lasso(data):
    X,y=read_data()
    m, n = X.shape
    weight = np.array([ 1.49462254e+01, -2.50275342e-01, -8.76423816e-03,  1.23727270e+00,
       -1.80224871e+02, -2.10165019e+02])
    max_iterations = 100000
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y;
