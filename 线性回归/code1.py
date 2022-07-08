import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from code import gradient_descent

# 多变量线性回归
# 第一列是房屋大小，第二列是卧室数量，第三列是房屋售价
# 根据已有数据，建立模型，预测房屋的售价

path = 'data1.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

# 1 特征归一化
# 观察数据发现，size变量是bedrooms变量的1000倍大小,统一量级会让梯度下降收敛的更快
data = (data - data.mean()) / data.std()
# print(data.head())

# 2 梯度下降
# 加一列常数项
data.insert(0, 'Ones', 1)
# 初始化X和y
cols = data.shape[1]
X2 = data.iloc[:, 0:cols - 1]
y2 = data.iloc[:, cols - 1:cols]
# 转换成matrix格式，初始化theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))

# 初始化学习速率和迭代次数
alpha = 0.01
iters = 1500
# 运行梯度下降算法
g2, cost2 = gradient_descent(X2, y2, theta2, alpha, iters)
# print(g2)