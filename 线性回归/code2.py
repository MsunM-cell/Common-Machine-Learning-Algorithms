import numpy as np
import pandas as pd
from sklearn import linear_model

path = 'data.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.insert(0, 'Ones', 1)

X = data.iloc[:, :-1].values
y = data['Profit'].values

# create linear regression object
linear = linear_model.LinearRegression()

# train the model using the training sets and check score
linear.fit(X, y)
linear.score(X, y)

# equation coefficient and intercept
print('coefficient: \n', linear.coef_)
print('intercept: \n', linear.intercept_)

X1 = [[1, 3.5], [1, 7]]
# predict output
predicted = linear.predict(X1)
print(predicted)
