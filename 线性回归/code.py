import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1 输出一个5*5的单位矩阵
A = np.eye(5)
# print(A)

# 2 单变量的线性回归
# 根据城市人口数量，预测开小吃店的利润

# 2.1 Plotting the data
path = 'data.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# print(data.head())
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))


# plt.show()

# 2.2 梯度下降
# 训练线性回归的参数θ

# 计算代价的函数
def compute_cost(X1, y1, theta1):
    inner = np.power(((X1 * theta1.T) - y1), 2)
    return np.sum(inner) / (2 * len(X1))


# 加入一列x，用于更新θ0
data.insert(0, 'Ones', 1)
# 初始化X和y
cols = data.shape[1]
X = data.iloc[:, :-1]
y = data.iloc[:, cols - 1:cols]
# print(X.head())
# print(y.head())
# 转换成numpy矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)
# 初始化theta
theta = np.matrix(np.array([0, 0]))

# 计算代价
J = compute_cost(X, y, theta)
# print(J)

# 初始化学习速率和迭代次数
alpha = 0.01
iters = 1500


# 梯度下降的函数
# 这个部分实现了Ѳ的更新
def gradient_descent(X1, y1, theta1, alpha1, iters1):
    temp = np.matrix(np.zeros(theta1.shape))
    parameters = int(theta1.ravel().shape[1])
    cost = np.zeros(iters1)

    for i in range(iters1):
        error = (X1 * theta1.T) - y1

        for j in range(parameters):
            term = np.multiply(error, X1[:, j])
            temp[0, j] = theta1[0, j] - ((alpha1 / len(X)) * np.sum(term))

        theta1 = temp
        cost[i] = compute_cost(X1, y1, theta1)

    return theta1, cost


g, cost = gradient_descent(X, y, theta, alpha, iters)
# print(g)

# 预测35000和70000城市规模的小吃摊利润
predict1 = [1, 3.5] * g.T
# print("predict1:", predict1)
predict2 = [1, 7] * g.T
# print("predict2:", predict2)

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# plt.show()

# 补充：正规方程
# 正规方程的函数
def normal_eqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # X.T@X等价于X.T.dot(X)
    return theta


final_theta = normal_eqn(X, y)
print(final_theta)
