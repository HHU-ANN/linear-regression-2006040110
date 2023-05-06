# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np




def ridge(data):
    X,y=read_data()
    z=np.matmul(X.T,X)+np.eye(X.shape[1])*(0.000000000000000000000000000001)
    w=np.matmul(np.linalg.inv(z),np.matmul(X.T,y))
    return w @ data
    
def lasso(data):  
    lr = 1e-10 
    epoch = 10000
    alpha = 0.1
    X, y = read_data()
    w = np.zeros(X.shape[1])
    for i in range(epoch):
        gradient = np.dot(X.T, (np.dot(X, w) - y)) + alpha * np.sign(w)
        w-= lr * gradient

    return w @ data


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y;
