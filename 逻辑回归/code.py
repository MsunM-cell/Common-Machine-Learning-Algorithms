import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = 'data.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
# print(data.head())

# 数据可视化
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]


# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')


# 实现sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 实现代价函数
def cost(theta1, X1, y1):
    theta1 = np.matrix(theta1)
    X1 = np.matrix(X1)
    y1 = np.matrix(y1)
    first = np.multiply(-y1, np.log(sigmoid(X1 * theta1.T)))
    second = np.multiply((1 - y1), np.log(1 - sigmoid(X1 * theta1.T)))
    return np.sum(first - second) / (len(X))


# 初始化X, y, theta
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]
theta = np.zeros(3)

# 转换X，y的类型
X = np.array(X.values)
y = np.array(y.values)

# 计算代价
J = cost(theta, X, y)


# 实现梯度计算的函数（并没有更新theta）
def gradient(theta1, X1, y1):
    theta1 = np.matrix(theta1)
    X1 = np.matrix(X1)
    y1 = np.matrix(y1)

    parameters = int(theta1.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X1 * theta1.T) - y1

    for i in range(parameters):
        term = np.multiply(error, X1[:, i])
        grad[i] = np.sum(term) / len(X1)

    return grad


res = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
J = cost(res[0], X, y)
# print(J)

# 画出决策曲线
plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = (-res[0][0] - res[0][1] * plotting_x1) / res[0][2]

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(plotting_x1, plotting_h1, 'y', label='Prediction')
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')


# plt.show()

# 评价逻辑回归模型
def hfunc1(theta1, X1):
    return sigmoid(np.dot(theta1.T, X1))


p = hfunc1(res[0], [1, 45, 85])
print(p)


# 定义预测函数
def predict(theta1, X1):
    probability = sigmoid(X1 * theta1.T)
    return [1 if x >= 0.5 else 0 for x in probability]


# 统计预测正确率
theta_min = np.matrix(res[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
