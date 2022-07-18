import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from code import sigmoid, predict
import scipy.optimize as opt

# 2 正则化逻辑回归

# 2.1 数据可视化

path = 'data1.txt'
data_init = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
# print(data_init.head())

positive = data_init[data_init['Accepted'].isin([1])]
negative = data_init[data_init['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
# plt.show()

# 2.2 特征映射

degree = 6
data = data_init
x1 = data['Test 1']
x2 = data['Test 2']

data.insert(3, 'Ones', 1)

for i in range(1, degree + 1):
    for j in range(0, i + 1):
        data['F' + str(i - j) + str(j)] = np.power(x1, i - j) * np.power(x2, j)

data.drop('Test 1', axis=1, inplace=True)
data.drop('Test 2', axis=1, inplace=True)


# print(data.head())

# 2.3 代价函数和梯度

# 实现正则化的代价函数
def cost_reg(theta, X, y, lr):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (lr / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


# 实现正则化的梯度函数
def gradient_reg(theta, X, y, lr):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if i == 0:
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((lr / len(X)) * theta[:, i])

    return grad


# 初始化X，y，theta
cols = data.shape[1]
X = data.iloc[:, 1:cols]
y = data.iloc[:, 0:1]
theta = np.zeros(cols - 1)

# 进行类型转换
X = np.array(X.values)
y = np.array(y.values)

# λ设为1
learning_rate = 1

# 计算初始代价
cost = cost_reg(theta, X, y, learning_rate)
# print(cost)

# 2.3.1 用工具库求解参数

res = opt.fmin_tnc(func=cost_reg, x0=theta, fprime=gradient_reg, args=(X, y, learning_rate))
# print(res)

# 使用预测函数来查看我们的方案在训练数据上的准确度
theta_min = np.matrix(res[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))


# print('accuracy = {0}%'.format(accuracy))

# 2.4 画出决策的曲线

def hfunc(theta, x1, x2):
    temp = theta[0][0]
    place = 0
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            temp += np.power(x1, i - j) * np.power(x2, j) * theta[0][place + 1]
            place += 1
    return temp


def find_decision_boundary(theta):
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    h_val = pd.DataFrame({'x1': x_cord, 'x2': y_cord})
    h_val['hval'] = hfunc(theta, h_val['x1'], h_val['x2'])

    decision = h_val[np.abs(h_val['hval']) < 2 * 10 ** -3]
    return decision.x1, decision.x2


# x, y = find_decision_boundary(res)
# plt.scatter(x, y, c='y', s=10, label='Prediction')
# plt.show()

# 2.5 改变λ，观察决策曲线

# λ=0时，过拟合

learning_rate_1 = 0
res_1 = opt.fmin_tnc(func=cost_reg, x0=theta, fprime=gradient_reg, args=(X, y, learning_rate_1))

# x, y = find_decision_boundary(res_1)
# plt.scatter(x, y, c='y', s=10, label='Prediction')
# plt.show()

# λ=100时，欠拟合
learning_rate_2 = 100
res_2 = opt.fmin_tnc(func=cost_reg, x0=theta, fprime=gradient_reg, args=(X, y, learning_rate_2))

x, y = find_decision_boundary(res_2)
plt.scatter(x, y, c='y', s=10, label='Prediction')
plt.show()


