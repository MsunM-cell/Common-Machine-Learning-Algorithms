## 逻辑回归（Logistic Regression）
- `code.py`手推逻辑回归，包括`sigmoid`函数、代价函数、梯度函数等的手动实现；基于线性分割，通过申请学生两次测试的评分，来决定他们是否被录取。
- `code1.py`手推正则化逻辑回归，包括特征映射、代价函数、梯度函数等的手动实现；引入正则项提升逻辑回归算法，根据一些芯片在两次测试中的测试结果，决定是否芯片要被接受或抛弃。
- `code2.py`调用`scikit-learn`机器学习库中的`linear_model`模块，通过`LogisticRegression`训练模型。